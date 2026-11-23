# SPDX-License-Identifier: GPL-3.0-only
"""Shared Gemini camera, bounded hand inference and synchronized live rendering."""
import base64
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time
import uuid
import numpy as np
from PyQt5 import QtCore,QtGui,QtWidgets
from studio.cameras.gui_camera import select_camera_intrinsics
from overlay_view import OverlayView
ROOT=Path(__file__).resolve().parents[1]


class HandLivePanel(QtWidgets.QWidget):
    def __init__(self,hand,canvas_factory,viewer_factory):
        super().__init__();self.hand=hand;self.host=hand.host;self.active=False;self.pending=None;self.paint_pending=None;self.temp=None;self.warming=False
        self.samples=deque(maxlen=2000);self.last_summary=None;self.last_index=None;self.next_due=0;self.last_image=None
        self.render_waiting=None;self.render_dropped=0
        self.timer=QtCore.QTimer(self);self.timer.setInterval(5);self.timer.setTimerType(QtCore.Qt.PreciseTimer);self.timer.timeout.connect(self.poll)
        layout=QtWidgets.QVBoxLayout(self)
        self.camera_source=QtWidgets.QComboBox()
        self.camera_source.setModel(self.host.camera_backend.model())
        self.camera_source.setCurrentIndex(self.host.camera_backend.currentIndex())
        self.camera_source.currentIndexChanged.connect(self.host.camera_backend.setCurrentIndex)
        self.host.camera_backend.currentIndexChanged.connect(self.camera_source.setCurrentIndex)
        layout.addWidget(self.camera_source)
        note=QtWidgets.QLabel('① 在上方应用方案，自动用本地照片预热 → ② 打开相机 → ③ 开始实时重建。无需在线摆手预热；下方保留补充预热。仅使用 RGB 与实时内参，不使用深度拟合。')
        note.setWordWrap(True);layout.addWidget(note)
        row=QtWidgets.QHBoxLayout();layout.addLayout(row)
        self.open=QtWidgets.QPushButton('打开相机预览');self.open.clicked.connect(self.open_camera);row.addWidget(self.open)
        self.close=QtWidgets.QPushButton('关闭相机');self.close.clicked.connect(self.close_camera);row.addWidget(self.close)
        self.warm=QtWidgets.QPushButton('在线补充预热（可选）');self.warm.clicked.connect(self.warmup);row.addWidget(self.warm)
        self.start=QtWidgets.QPushButton('开始实时重建');self.start.setObjectName('primary');self.start.clicked.connect(self.begin);row.addWidget(self.start)
        self.stop=QtWidgets.QPushButton('停止实时重建');self.stop.clicked.connect(lambda:self.end('实时重建已停止，保留最后结果'));row.addWidget(self.stop)
        row.addWidget(QtWidgets.QLabel('显示'))
        self.view=QtWidgets.QComboBox();self.view.addItem('原图叠加（推荐）','overlay');self.view.addItem('三维场景','3d');row.addWidget(self.view)
        row.addWidget(QtWidgets.QLabel('上限 FPS'))
        self.limit=QtWidgets.QSpinBox();self.limit.setRange(1,30);self.limit.setValue(15);row.addWidget(self.limit)
        self.status=QtWidgets.QLabel('相机未打开。实时流不保存视频、RGB 或逐帧重建，只保留停止后的计时摘要。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.latency=QtWidgets.QLabel('延迟拆分将在实时重建后显示；将鼠标悬停于状态行可查看各阶段计时。');self.latency.setWordWrap(True);self.latency.setObjectName('muted');layout.addWidget(self.latency)
        self.split=QtWidgets.QSplitter();layout.addWidget(self.split,1)
        left=QtWidgets.QWidget();box=QtWidgets.QVBoxLayout(left);box.setContentsMargins(0,0,0,0)
        box.addWidget(QtWidgets.QLabel('最新彩色预览 · 请让整只手和少量手腕入画'))
        self.preview=canvas_factory(selectable=False);self.preview.setMinimumSize(240,180);box.addWidget(self.preview,1);self.split.addWidget(left)
        self.stack=QtWidgets.QStackedWidget();self.split.addWidget(self.stack);self.split.setSizes([330,900])
        self.overlay=OverlayView();self.overlay.set_layers(True,False,False);self.stack.addWidget(self.overlay)
        self.viewer=viewer_factory();self.viewer.grid.setSize(.7,.7);self.viewer.grid.setSpacing(.05,.05);self.stack.addWidget(self.viewer)
        self.view.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.overlay.frameSwapped.connect(lambda:self.painted('overlay'));self.viewer.frameSwapped.connect(lambda:self.painted('3d'))
        self.overlay.renderFailed.connect(lambda message:self.end('叠加渲染失败：'+message))
        self.host.camera.frame.connect(self.camera_frame);self.host.camera.messageReceived.connect(self.camera_event);self.host.camera.stopped.connect(self.camera_stopped)
        self.refresh()

    @property
    def running(self):return self.active or self.pending is not None or self.paint_pending is not None or self.warming

    def refresh(self):
        h=self.hand;w=self.host;ready=w.camera_state=='preview';running=self.running
        blocked=getattr(getattr(w,'live',None),'running',False) or w.inference_busy or w.rgbd.mode is not None or w.rgbd.pending_camera_action is not None or w.rgbd.batch.active or w.image_seg.active
        self.camera_source.setEnabled(not w.camera.active and not blocked and not running)
        self.open.setEnabled(not w.camera.active and w.camera_state in ('off','photo') and not blocked)
        self.close.setEnabled(w.camera.active and w.camera_state not in ('stopping',))
        self.warm.setEnabled(ready and h.state=='ready' and not running and not blocked)
        self.start.setEnabled(ready and h.state=='ready' and not running and not blocked)
        self.stop.setEnabled(self.active or self.warming)
        self.view.setEnabled(not running);self.limit.setEnabled(not running)

    def open_camera(self):
        if self.host.camera.active:return
        self.status.setText('正在连接 '+self.host.camera_backend.currentText()+'…');self.host.open_camera();self.refresh()

    def close_camera(self):
        self.end('相机关闭，实时重建已停止')
        if self.host.camera.active:self.host.close_camera()
        self.refresh()

    def camera_event(self,event):
        if event['type']=='ready':
            m=event['metadata'];warm=getattr(self.hand,'startup_warmup',None) or {}
            suffix='离线预热已完成，可直接开始。' if warm.get('status')=='ready' else '如未离线预热，可在线补充预热。'
            self.status.setText(f"{m['device']} · {m['width']}×{m['height']} · 采集 {m['fps']} FPS / 预览默认≤15 FPS，实时推理供帧≤30 FPS · 已读取实际内参。"+suffix)
        elif event['type']=='error':self.end('相机错误：'+event['message']);self.status.setText('相机错误：'+event['message'])
        self.refresh()

    def camera_stopped(self):
        self.end('相机已断开，实时重建已停止');self.refresh()

    def camera_frame(self,image):
        if self.host.camera_state in ('starting','preview','countdown'):
            self.preview.set_preview(image);self.poll()

    def prepare(self):
        if self.hand.state!='ready' or self.host.camera_state!='preview' or self.running:return False
        if getattr(getattr(self.host,'live',None),'running',False) or self.host.inference_busy or self.host.rgbd.mode is not None:return False
        self.hand.sequence.timer.stop();self.temp=tempfile.TemporaryDirectory(prefix='sam3d-hand-live-')
        self.stream_id=uuid.uuid4().hex;self.last_index=None;self.next_due=0;self.started=time.monotonic();self.last_summary=None
        self.samples.clear();self.attempts=self.successes=self.errors=0;self.result_count=0;self.viewer._hand_active=False
        self.render_waiting=None;self.render_dropped=0
        self.mode=self.view.currentData();self.confidence=self.hand.confidence.value();self.hand_selection=self.hand.hand_selection.currentData();self.padding=1.1
        self.execution=self.hand.execution_config();self.metadata=dict(self.host.capture_metadata or {})
        self.summary_path=None;self.runtime=None;return True

    def begin(self):
        if not self.prepare():return
        if self.mode=='overlay' and self.overlay.failed:self.abort('叠加渲染器不可用，请改用三维场景');return
        self.host.camera.send('preview_rate',fps=30)
        self.active=True;self.timer.start();self.status.setText('等待最新帧；繁忙时丢弃旧帧，不排队。首次出现未预热批量可能较慢。')
        self.hand.update_controls();self.host.update_camera_controls();self.poll()

    def warmup(self):
        if not self.prepare():return
        self.host.camera.send('preview_rate',fps=30)
        self.warming=True;self.timer.start();self.status.setText('正在预热，首次编译可能需要较长时间；请将手放在画面内…')
        self.hand.update_controls();self.host.update_camera_controls();self.poll()

    def poll(self):
        now=time.monotonic()
        if self.pending:
            if now-self.sent_at>(180 if self.warming else 60):self.hand.shutdown();self.status.setText('实时推理超时，模型已释放')
            return
        if not (self.active or self.warming) or now<self.next_due:return
        if not self.isVisible() or self.window().isMinimized():return
        w=self.host;p=w.camera.latest_frame
        if w.camera_state!='preview':self.end('相机已停止预览');return
        if not p or p['index']==self.last_index or now-p.get('received_monotonic',0)>.5:return
        try:
            encoded=p.get('_jpeg_bytes')
            if encoded is None:encoded=base64.b64decode(p['jpeg'],validate=True)
            image=p.get('_image')
            if image is None:image=QtGui.QImage.fromData(encoded,'JPG')
            if image.isNull():raise ValueError('相机图像无效')
            intr,_=select_camera_intrinsics(True,image.width(),image.height(),w.capture_metadata)
            if intr is None:raise ValueError('相机没有可用实时内参')
            matrix=[[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]]
            self.client_received_at=p.get('client_receive_monotonic',p['received_monotonic']);self.client_decode_ms=p.get('client_decode_ms',0)
            self.gui_delivery_ms=(p.get('gui_receive_monotonic',now)-p.get('decoded_monotonic',now))*1000
            self.last_image=image;self.received_at=p['received_monotonic'];self.sent_at=now;self.last_index=p['index'];self.attempts+=1
            source=Path(self.temp.name)/'input.jpg';source.write_bytes(encoded)
            self.pending=uuid.uuid4().hex;self.hand.job=self.pending;self.hand.state='busy'
            self.confidence=self.hand.confidence.value()
            self.hand_selection=self.hand.hand_selection.currentData()
            cmd=dict(type='hand_stream',job=self.pending,stream_id=self.stream_id,image=str(source),frame_index=p['index'],received_monotonic=self.received_at,output=self.temp.name,camera_matrix=matrix,confidence=self.confidence,hand_selection=self.hand_selection,padding=self.padding,warmup=self.warming,compact_result=True,one_euro=self.hand.one_euro_options())
            payload=(json.dumps(cmd)+'\n').encode();self.queued_at=time.monotonic()
            if self.hand.process.write(payload)<0:raise RuntimeError('模型进程不可写')
            if self.hand.process.bytesToWrite():self.hand.process.waitForBytesWritten(1)
            self.next_due=now+1/self.limit.value()
        except Exception as exc:
            self.pending=None;self.hand.job=None
            if self.hand.state=='busy':self.hand.state='ready'
            self.abort('无法发送相机帧：'+str(exc))

    def handle_event(self,event):
        if not self.pending or event.get('job')!=self.pending or event['type'] not in ('hand_stream_result','error'):return False
        event_received=time.monotonic()
        self.pending=None;self.hand.job=None;self.hand.state='ready'
        if not (self.active or self.warming):self.finish();return True
        if event['type']=='error':
            self.errors+=1;self.status.setText('当前帧失败：'+event['message']+'；将重试最新帧')
            self.render_dropped+=int(self.paint_pending is not None)+int(self.render_waiting is not None)
            self.paint_pending=self.render_waiting=None
            self.viewer.clear_mesh();self.viewer._hand_active=False;self.overlay.clear_frame(self.last_image)
            if self.warming:self.end('预热失败：'+event['message'])
            else:self.next_due=time.monotonic()+.2
        else:
            try:
                report=event['report'];self.result_count+=1
                if self.result_count==1:self.hand.info.setText('手部模型已就绪；相机实时耗时与状态见实时页。')
                if report.get('hand_reports'):
                    first=report['hand_reports'][0]
                    self.runtime={key:first.get(key) for key in ('execution','backbone','backbone_runtime','backbone_engine_executions','execution_counts','feedback_runtime')}
                    self.runtime['detector']=report.get('detector')
                if self.runtime and report.get('frame_execution'):
                    execution=report['frame_execution']
                    self.runtime.update({k:v for k,v in execution.items() if k!='total_backbone_executions'})
                    if 'total_backbone_executions' in execution:
                        self.runtime['backbone_runtime']['executions']=execution['total_backbone_executions']
                scene_open=time.monotonic()
                with np.load(event['path'],allow_pickle=False) as archive:
                    data={key:archive[key] for key in ('vertices','faces','joints','edges','camera_matrix')}
                scene_loaded=time.monotonic()
                if self.warming:
                    self.hand.display_scene(self.last_image,data,report,self.viewer,self.overlay,continuous=True,mode=self.mode)
                    batches=report.get('warmup_batch_sizes',[])
                    self.end('单手 / 双手预热完成，可以开始实时重建' if batches else '未检测到手：请让整只手入画后重新预热')
                else:
                    metrics=dict(one_euro_ms=report.get('one_euro',{}).get('processing_ms',0.),one_euro=report.get('one_euro',{}),detection_options=report.get('detection_options',{}),frame_age_at_dispatch_ms=(self.sent_at-self.received_at)*1000,camera_delivery_ms=(self.client_received_at-self.received_at)*1000,latest_frame_wait_ms=(self.sent_at-self.client_received_at)*1000,client_decode_ms=self.client_decode_ms,gui_delivery_ms=self.gui_delivery_ms,command_preparation_ms=(self.queued_at-self.sent_at)*1000,worker_queue_ms=(event.get('worker_start_monotonic',self.queued_at)-self.queued_at)*1000,result_delivery_ms=(event_received-event.get('worker_finished_monotonic',event_received))*1000,scene_read_ms=(scene_loaded-scene_open)*1000,**event.get('stage_timings',{}))
                    metrics.update({'detector_'+k+'_ms':v for k,v in report.get('detector_timings',{}).items()})
                    if event['frame_index']!=self.last_index:raise ValueError('实时结果帧编号与 RGB 不匹配')
                    sample=dict(**metrics,detected_hands=report.get('hands',[]),frame_index=event['frame_index'],received_monotonic=self.received_at,dispatch_monotonic=self.sent_at,result_monotonic=time.monotonic(),worker_ms=event['worker_ms'],detect_ms=report['detect_ms'],reconstruct_ms=report['reconstruct_ms'],hands=report['hand_count'])
                    payload=(self.last_image,data,report,sample)
                    if self.paint_pending:
                        if self.render_waiting is not None:self.render_dropped+=1
                        self.render_waiting=payload
                    else:self.submit_render(payload)
                    if not report['hand_count']:self.next_due=max(self.next_due,time.monotonic()+.1)
                    # Rendering owns its RGB/mesh snapshot. Next inference no
                    # longer waits for top-level composition / frameSwapped.
                    QtCore.QTimer.singleShot(0,self.poll)
            except Exception as exc:self.abort('读取实时结果失败：'+str(exc))
        # The whole run already locks photo/sequence controls. Avoid toggling
        # dozens of styled Qt widgets busy/ready for every incoming frame.
        if not self.running:self.hand.update_controls()
        return True

    def submit_render(self,payload):
        image,data,report,sample=payload
        began=time.monotonic()
        sample['render_queue_ms']=(began-sample['result_monotonic'])*1000
        self.hand.display_scene(image,data,report,self.viewer,self.overlay,continuous=True,mode=self.mode)
        if self.mode=='overlay':self.overlay.frame_index=sample['frame_index']
        sample['scene_apply_ms']=(time.monotonic()-began)*1000
        self.paint_pending=sample
        (self.overlay if self.mode=='overlay' else self.viewer).update()

    def painted(self,mode):
        if not self.paint_pending or mode!=self.mode:return
        if mode=='overlay' and self.overlay.painted_frame_index!=self.paint_pending['frame_index']:return
        sample=self.paint_pending;self.paint_pending=None;now=time.monotonic()
        if mode=='overlay':sample['paint_cpu_ms']=self.overlay.paint_cpu_ms
        sample.update(display_monotonic=now,frame_to_display_ms=(now-sample['received_monotonic'])*1000,dispatch_to_display_ms=(now-sample['dispatch_monotonic'])*1000,result_to_display_ms=(now-sample['result_monotonic'])*1000)
        self.samples.append(sample);self.successes+=1
        recent=list(self.samples)[-30:];fps=(len(recent)-1)/(now-recent[0]['display_monotonic']) if len(recent)>1 else 0
        transport=sample['dispatch_to_display_ms']-sample['worker_ms']-sample['result_to_display_ms']
        self.latency.setText(f"延迟拆分：取帧时已旧 {sample['frame_age_at_dispatch_ms']:.0f} ms ＋ 工作进程 {sample['worker_ms']:.0f} ms ＋ 提交 / 传输 / 装载 {max(0,transport):.0f} ms ＋ 等待显示 {sample['result_to_display_ms']:.0f} ms。非相机曝光到屏幕的完整物理延迟。")
        self.status.setToolTip(f"One Euro：{sample.get('one_euro',{}).get('processing_ms',0):.2f} ms\n"+'帧到显示拆分：\n'+ '\n'.join(f'{label}：{sample.get(key,0):.1f} ms' for label,key in [('派发时帧龄','frame_age_at_dispatch_ms'),('相机到接收线程','camera_delivery_ms'),('JPEG 解码','client_decode_ms'),('解码完成到 GUI','gui_delivery_ms'),('等待取用最新帧（含解码）','latest_frame_wait_ms'),('提交准备','command_preparation_ms'),('工作进程排队','worker_queue_ms'),('工作进程处理','worker_ms'),('结果送达','result_delivery_ms'),('读取场景','scene_read_ms'),('等待提交显示','render_queue_ms'),('更新场景','scene_apply_ms'),('绘制 CPU 调用','paint_cpu_ms'),('结果到显示','result_to_display_ms')])+f'\n待显示结果丢弃：{self.render_dropped} 帧；分项可能包含关系，不能重复相加。')
        self.status.setText(f"{sample['hands']} 手 · 显示 {fps:.1f} FPS · 检测 {sample['detect_ms']:.1f} ms / 重建 {sample['reconstruct_ms']:.1f} ms · 帧到显示 {sample['frame_to_display_ms']:.0f} ms · 只取最新帧 / 推理显示并行"+(f" · One Euro {sample.get('one_euro',{}).get('processing_ms',0):.1f} ms" if sample.get('one_euro',{}).get('pose') or sample.get('one_euro',{}).get('position') else '')+(' · 未检测到手，已清除旧网格' if not sample['hands'] else ''))
        if self.render_waiting is not None and self.active:
            payload,self.render_waiting=self.render_waiting,None;self.submit_render(payload)
        if not self.active and not self.pending:self.finish()
        elif self.active:QtCore.QTimer.singleShot(0,self.poll)

    def end(self,message):
        if not self.running:return
        self.active=False;self.warming=False;self.stop_message=message;self.paint_pending=None;self.render_waiting=None
        self.status.setText('正在等待当前帧完成…' if self.pending else message)
        if not self.pending:self.finish()
        self.hand.update_controls()

    def abort(self,message):
        self.active=self.warming=False;self.pending=self.paint_pending=None;self.render_waiting=None;self.stop_message=message
        self.finish();self.status.setText(message)

    def finish(self):
        self.timer.stop()
        if self.temp:
            samples=list(self.samples);summary=dict(execution=self.execution,runtime=self.runtime,detection_options=dict(confidence=self.confidence,hand_selection=self.hand_selection),camera=self.metadata,display_mode=self.mode,attempts=self.attempts,displayed=self.successes,errors=self.errors,render_dropped=self.render_dropped,scheduling='inference/display overlap; one in flight + one waiting render',samples=samples,duration_seconds=time.monotonic()-self.started,timing_scope='Host SDK color receipt to OpenGL frameSwapped; excludes sensor exposure and USB latency; latest frame, no depth fitting')
            if len(samples)>1:summary['display_fps']=(len(samples)-1)/(samples[-1]['display_monotonic']-samples[0]['display_monotonic'])
            for key in ('one_euro_ms','detect_ms','reconstruct_ms','worker_ms','frame_to_display_ms','dispatch_to_display_ms','result_to_display_ms','frame_age_at_dispatch_ms','camera_delivery_ms','latest_frame_wait_ms','client_decode_ms','gui_delivery_ms','render_queue_ms','paint_cpu_ms','command_preparation_ms','worker_queue_ms','result_delivery_ms','read_decode_ms','inference_wall_ms','scene_write_ms','scene_read_ms','scene_apply_ms','detector_preprocess_ms','detector_inference_ms','detector_postprocess_ms','detector_other_ms'):
                values=[s[key] for s in samples if key in s]
                if values:summary[key]=dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)))
            self.last_summary=summary
            try:
                directory=ROOT/'outputs/hand-streams';directory.mkdir(parents=True,exist_ok=True)
                self.summary_path=directory/(datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.json')
                self.summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
            except OSError as exc:self.stop_message=getattr(self,'stop_message','实时重建已停止')+'；摘要未保存：'+str(exc)
            finally:self.temp.cleanup();self.temp=None

        self.host.camera.send('preview_rate',fps=15)
        self.last_image=None
        self.status.setText(getattr(self,'stop_message','实时重建已停止'));self.hand.update_controls()
        if hasattr(self.host,'live'):self.host.update_camera_controls()
