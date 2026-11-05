# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Pose-set UI and serial batch orchestration around the existing RGB-D workspace."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from PyQt5 import QtCore,QtGui,QtWidgets
from rgbd_batch import PoseSet,digest

ROOT=Path(__file__).resolve().parents[1]
def stamp():return datetime.now().strftime('%Y%m%d-%H%M%S-%f')

class PoseSetPanel(QtWidgets.QGroupBox):
    def __init__(self,workspace):
        super().__init__('多姿态样本集（可选）');self.v=workspace;self.host=workspace.host
        self.session=None;self.job=None;self.capture_index=None;self.driving=False
        layout=QtWidgets.QVBoxLayout(self);layout.setContentsMargins(6,6,6,6)
        row=QtWidgets.QHBoxLayout();row.addWidget(QtWidgets.QLabel('照片数量'))
        self.count=QtWidgets.QSpinBox();self.count.setRange(1,100);self.count.setValue(5);row.addWidget(self.count)
        self.new=QtWidgets.QPushButton('创建姿态1～N');self.new.clicked.connect(self.new_set);row.addWidget(self.new)
        self.open=QtWidgets.QPushButton('打开样本集');self.open.clicked.connect(self.open_set);row.addWidget(self.open)
        self.leave=QtWidgets.QPushButton('退出样本集');self.leave.clicked.connect(self.leave_set);row.addWidget(self.leave)
        self.poses=QtWidgets.QComboBox();self.poses.setMinimumWidth(180);self.poses.setIconSize(QtCore.QSize(64,36));self.poses.currentIndexChanged.connect(self.select);row.addWidget(self.poses,1)
        self.next=QtWidgets.QPushButton('确认，下一姿态');self.next.clicked.connect(self.next_pose);row.addWidget(self.next)
        layout.addLayout(row)
        row=QtWidgets.QHBoxLayout()
        self.segment=QtWidgets.QPushButton('SAM3 分割全部姿态');self.segment.clicked.connect(self.segment_all);row.addWidget(self.segment)
        self.run=QtWidgets.QPushButton('当前方案运行全部姿态');self.run.clicked.connect(self.run_all);row.addWidget(self.run)
        self.resume=QtWidgets.QCheckBox('跳过已完成');self.resume.setChecked(True);row.addWidget(self.resume)
        self.compare=QtWidgets.QPushButton('联合对照');self.compare.clicked.connect(self.show_comparison);row.addWidget(self.compare)
        self.export=QtWidgets.QPushButton('导出整组');self.export.clicked.connect(self.export_set);row.addWidget(self.export)
        layout.addLayout(row)
        self.note=QtWidgets.QLabel('设置照片数量，创建样本集；也可继续使用下方单张流程。');self.note.setWordWrap(True);layout.addWidget(self.note)
        self.timer=QtCore.QTimer(self);self.timer.setInterval(150);self.timer.timeout.connect(self.tick);self.timer.start()
        self.host.reconstructionFinished.connect(self.reconstructed)
        self.refresh()
    @property
    def active(self):return self.job is not None
    def idle(self):
        v=self.v;h=self.host
        return not self.active and not v.segmenting and v.process.state()==QtCore.QProcess.NotRunning and v.expected is None and v.pending_camera_action is None and not h.inference_busy and not h.camera.active and not (getattr(h,'live',None) and h.live.running)
    def message(self,text):self.note.setText(text)
    def new_set(self):
        if not self.idle():return
        session=PoseSet.create(ROOT/'outputs/rgbd-sets'/stamp(),self.count.value());self.set_session(session)
    def open_set(self):
        if not self.idle():return
        directory=QtWidgets.QFileDialog.getExistingDirectory(self,'打开包含 pose_set.json 的样本集',str(ROOT/'outputs/rgbd-sets'))
        if directory:
            try:self.set_session(PoseSet(directory))
            except Exception as exc:self.message('打开失败：'+str(exc))
    def set_session(self,session):
        self.session=session;self.capture_index=None
        self.count.setValue(len(session.poses))
        self.poses.blockSignals(True);self.poses.clear()
        for p in session.poses:self.poses.addItem(p['name'])
        self.poses.blockSignals(False);self.select(0);self.refresh()
        self.message(f'请依次拍摄 {len(session.poses)} 张不同姿态照片；当前为姿态1。每张可重拍，确认后进入下一姿态。')
    def leave_set(self):
        if not self.idle():return
        self.session=None;self.poses.clear();self.capture_index=None;self.refresh();self.message('已退出样本集；磁盘数据保留。')
    def select(self,index):
        if not self.session or index<0 or self.active:return
        if self.v.process.state()!=QtCore.QProcess.NotRunning or self.v.pending_camera_action is not None:return
        try:
            path=self.session.sample_path(self.session.poses[index])
            if path:self.v.load_sample(path)
            else:self.v.clear_sample()
            self.message(f"{self.session.poses[index]['name']}："+('可标前景/背景点、检查分割或重拍。' if path else '尚未拍摄。开启预览，摆好姿势后倒计时拍摄。'))
        except Exception as exc:self.message('姿态载入失败：'+str(exc))
    def next_pose(self):
        if not self.session or not self.idle():return
        i=self.poses.currentIndex()
        if not self.session.sample_path(self.session.poses[i]):self.message('请先拍摄当前姿态。');return
        if i+1<len(self.session.poses):
            self.poses.setCurrentIndex(i+1);self.v.toggle_preview()
        else:self.message('全部姿态已拍摄。请切换姿态逐张标点，再统一分割；也可直接批量重建。')
    def capture_directory(self):
        if not self.session:return None
        self.capture_index=self.poses.currentIndex()
        return self.session.directory/'samples'/f'pose-{self.capture_index+1:03d}'/stamp()
    def confirm_capture(self):
        if not self.session:return True
        i=self.poses.currentIndex()
        if self.session.sample_path(self.session.poses[i]):
            return QtWidgets.QMessageBox.question(self,'重拍当前姿态','重拍会将新照片设为当前姿态；旧照片和结果仍保留在样本集目录中。继续？')==QtWidgets.QMessageBox.Yes
        return True
    def captured(self,path):
        if self.session and self.capture_index is not None:
            i=self.capture_index;self.session.attach(i,path);self.capture_index=None;self.refresh()
            self.message(f'姿态{i+1}已保存。检查照片，满意后点击“确认，下一姿态”；全部拍完后逐张标点。')
    def refresh(self):
        busy=not self.idle();available=self.session is not None
        self.v.profile_btn.setEnabled(not self.active)
        self.v.target_mask.setEnabled(not busy)
        self.v.open_btn.setEnabled(not busy and not available)
        self.v.open_btn.setToolTip('退出样本集后可打开单张样本；整组请使用“打开样本集”。' if available else '')
        for w in (self.new,self.open,self.count):w.setEnabled(not busy)
        for w in (self.leave,self.poses,self.next,self.segment,self.run,self.compare,self.export,self.resume):w.setEnabled(available and not busy)
        if self.session:
            self.poses.blockSignals(True)
            for i,p in enumerate(self.session.poses):
                path=self.session.sample_path(p);status='待拍摄'
                if path:
                    status='已拍摄'
                    prompt=path/'sam3_prompts.json'
                    if prompt.is_file():
                        try:
                            labels=json.loads(prompt.read_text())['point_labels'];status=f'标点 {labels.count(1)}前/{labels.count(0)}后'
                        except (ValueError,KeyError):status='标点无效'
                    if (path/'mask.png').is_file():status+=' · 有掩膜'
                    self.poses.setItemIcon(i,QtGui.QIcon(str(path/'rgb.png')))
                self.poses.setItemText(i,p['name']+' · '+status)
            self.poses.blockSignals(False)
        if self.active and not self.driving:
            for w in (self.v.preview_btn,self.v.capture_btn,self.v.open_btn,self.v.run_btn,self.v.segment_btn,self.v.evaluate_btn,self.v.canvas,self.v.candidates,self.v.target_mask,self.v.profile_btn):w.setEnabled(False)
            for w in (self.v.fg_btn,self.v.bg_btn,self.v.undo_point,self.v.clear_points):w.setEnabled(False)
            self.v.cancel_btn.setEnabled(True)
    def paths(self):
        if not self.session:raise ValueError('请先创建或打开样本集')
        result=[]
        for i,p in enumerate(self.session.poses):
            path=self.session.sample_path(p)
            if not path:raise ValueError(p['name']+'尚未拍摄')
            if not (path/'rgbd.npz').is_file():raise ValueError(p['name']+'数据不完整')
            result.append((i,path))
        return result
    def segment_all(self):
        if not self.idle():return
        try:
            requests=[]
            for i,path in self.paths():
                file=path/'sam3_prompts.json'
                if not file.is_file():raise ValueError(f'请先为姿态{i+1}标注前景点和背景点')
                prompt=json.loads(file.read_text())
                if prompt.get('rgb_sha256')!=digest(path/'rgb.png') or 1 not in prompt.get('point_labels',[]):raise ValueError(f'姿态{i+1}缺少有效前景点，请重新标点')
                output=path/'segmentations'/stamp();output.parent.mkdir(exist_ok=True)
                requests.append(dict(image=str(path/'rgb.png'),rgb_sha256=prompt['rgb_sha256'],points_pixels=prompt['points_pixels'],point_labels=prompt['point_labels'],output=str(output)))
            run_dir=self.session.directory/'batch-segmentations'/stamp();run_dir.mkdir(parents=True)
            request=run_dir/'request.json';request.write_text(json.dumps(dict(requests=requests),ensure_ascii=False,indent=2))
            self.job=dict(kind='segment',phase='segment',outputs=[],errors=[],started=time.monotonic(),directory=str(run_dir))
            self.v.begin_segmentation(request,'segment_batch')
            self.message(f'SAM3 正在依次分割 {len(requests)} 个姿态；完成后再检查各张候选。');self.refresh()
        except Exception as exc:self.message(str(exc))
    def worker_event(self,event):
        if not self.active or self.job['kind']!='segment':return
        if event['type']=='segmented':
            self.job['outputs'].append(event);self.message(f"已分割 {len(self.job['outputs'])}/{len(self.session.poses)}；SAM3 尚未退出，请等待。")
        elif event['type']=='segment_error':self.job['errors'].append(event)
    def operation_finished(self,mode,ok):
        if mode=='preview':self.capture_index=None
        if mode=='capture':
            if ok and self.v.sample==self.v.pending_sample:self.captured(self.v.sample)
            else:self.capture_index=None
        if not self.active:return
        if mode=='segment_batch':
            job=self.job;self.job=None;applied=0
            for event in job['outputs']:
                try:
                    self.v.load_sample(Path(event['report']['image']).parent)
                    self.v.setup_candidates(event['output'],event['report']);self.v.choose_candidate();applied+=1
                except Exception as exc:job['errors'].append(dict(message=str(exc)))
            (Path(job['directory'])/'worker.log').write_text(self.v.log_text)
            self.session.data.setdefault('segment_runs',[]).append(dict(success=applied,total=len(self.session.poses),errors=job['errors'],process_ok=ok));self.session.save()
            self.poses.setCurrentIndex(0);self.select(0);self.v.refresh_controls()
            self.message(f'SAM3 已退出；已保存 {applied}/{len(self.session.poses)} 个掩膜，请逐张检查。'+(' 部分失败，可补点后重试。' if job['errors'] or not ok else ''))
        elif mode=='evaluate' and self.job['phase']=='evaluate':self.finish_item(ok,'评估失败' if not ok else '')
        self.refresh()
    def run_all(self):
        if not self.idle():return
        try:
            paths=self.paths();config=self.host.draft_config()
            if config is None:raise ValueError('请先设置有效启动方案')
            if config.one_euro_enabled or config.smpl_smoothing:raise ValueError('静态姿态集请先关闭 One Euro 和 MLP 平滑，避免不同姿态继承时间状态')
            selection='mask_bbox' if self.v.target_mask.isChecked() else 'detector'
            run=dict(id=stamp(),kind='reconstruct',status='running',config_id=config.config_id,config=config.to_dict(),selection=selection,items=[])
            self.session.data['runs'].append(run);self.session.save()
            self.job=dict(kind='reconstruct',phase='warmup',config=config,paths=paths,cursor=0,run=run,selection=selection,skip=self.resume.isChecked(),started=time.monotonic(),current=None)
            self.host.preload_model();self.host.update_camera_controls();self.message('正在应用并预热当前方案；就绪后依次处理全部姿态。');self.refresh()
        except Exception as exc:self.message(str(exc))
    def tick(self):
        if not self.active:return
        j=self.job
        if j['kind']!='reconstruct':
            if not self.v.segmenting and self.v.process.state()==QtCore.QProcess.NotRunning:self.operation_finished('segment_batch',False)
            return
        if self.host.draft_config()!=j['config'] or self.v.target_mask.isChecked()!=(j['selection']=='mask_bbox'):
            self.stop('配置已改变，批量任务已停止');return
        if self.host.model_failure:
            self.stop('模型失败：'+self.host.model_failure);return
        if time.monotonic()-j['started']>1200:self.stop('批量任务等待超时，已保留完成结果');return
        if j['phase']=='warmup':
            if self.host.model_state=='ready' and self.host.active_config==j['config']:j['phase']='next'
            elif self.host.model_state=='off' and not self.host.pending_launch:self.stop('模型未能启动，请检查第一页日志');return
        if j['phase']=='next':self.next_item()
    def next_item(self):
        j=self.job
        if self.v.mode is not None or self.host.inference_busy:return
        if j['cursor']>=len(j['paths']):
            j['run']['status']='complete';self.session.save();self.job=None;self.v.refresh_controls();self.host.update_camera_controls();self.refresh()
            counts={k:sum(r['status']==k for r in j['run']['items']) for k in ('success','skipped','failed')}
            self.message(f"批量完成：成功 {counts['success']}，复用 {counts['skipped']}，失败 {counts['failed']}。可切换方案再运行整组，或打开联合对照。")
            return
        index,path=j['paths'][j['cursor']];j['current']=index;j['started']=time.monotonic()
        if j['skip'] and self.session.latest(index,j['config'].config_id,j['selection']):self.finish_item(True,'',skipped=True);return
        try:
            self.driving=True;self.v.load_sample(path);self.poses.blockSignals(True);self.poses.setCurrentIndex(index);self.poses.blockSignals(False)
            j['phase']='infer';j['before']=len(self.v.results);j['fingerprint']=self.session.fingerprint(index)
            self.v.run_btn.setEnabled(True);self.v.reconstruct()
            if not self.host.inference_busy:self.finish_item(False,self.v.status.text())
            else:self.message(f"正在重建 姿态{index+1}/{len(j['paths'])} · {j['config'].summary()}")
        except Exception as exc:self.finish_item(False,str(exc))
        finally:self.driving=False;self.refresh()
    def reconstructed(self,ok):
        if self.active and self.job['kind']=='reconstruct' and self.job['phase']=='infer':
            job=self.job
            QtCore.QTimer.singleShot(0,lambda:self.after_reconstructed(ok) if self.job is job else None)
    def after_reconstructed(self,ok):
        if not self.active:return
        j=self.job
        if not ok or len(self.v.results)<=j['before']:self.finish_item(False,'人体重建失败');return
        result=self.v.results[-1];result.update(batch_config_id=j['config'].config_id,batch_input=j['fingerprint'],input_selection=j['selection'])
        self.v.save_results();j['result']=result
        if self.v.canvas.mask is not None:
            j['phase']='evaluate';self.message(f"正在评估 姿态{j['current']+1}/{len(j['paths'])}")
            if self.v.mode!='evaluate':self.finish_item(False,'评估未能启动')
        else:self.finish_item(True,'无掩膜，仅保留重建与耗时')
    def finish_item(self,ok,message,skipped=False):
        if not self.active:return
        j=self.job;j['run']['items'].append(dict(pose=j['current']+1,status='skipped' if skipped else ('success' if ok else 'failed'),message=message))
        self.session.save();j['cursor']+=1;j['phase']='next'
    def stop(self,message='已停止批量任务，完成结果保留，可再次运行续跑。'):
        if not self.active:return
        j=self.job;self.job=None
        if j['kind']=='reconstruct':j['run']['status']='interrupted';self.session.save()
        self.v.cancel();self.message(message);self.refresh();self.host.update_camera_controls()
        # Already saved candidates can only be applied AFTER the GPU process exits.
        if j['kind']=='segment':self.job=j
    def show_comparison(self):
        if not self.session or not self.idle():return
        dialog=QtWidgets.QDialog(self);dialog.setWindowTitle('姿态 × 启动方案 联合对照');dialog.resize(1050,650)
        layout=QtWidgets.QVBoxLayout(dialog);metric=QtWidgets.QComboBox()
        for label,key in [('推理 ms','inference_ms'),('深度 MAE mm','mae_mm'),('点面 MAE mm','point_surface_mae_mm'),('轮廓 IoU','silhouette_iou'),('覆盖率','model_coverage_fraction')]:metric.addItem(label,key)
        layout.addWidget(metric);table=QtWidgets.QTableWidget();layout.addWidget(table,1)
        note=QtWidgets.QLabel('每格取当前输入/掩膜下最后一次完成结果。点击格子载入三维场景；切换同一姿态的方案保留视角。');note.setWordWrap(True);layout.addWidget(note)
        configs={}
        for run in self.session.data['runs']:configs[(run['config_id'],run['selection'])]=run['config']
        keys=list(configs)
        def populate():
            table.clear();table.setRowCount(len(self.session.poses));table.setColumnCount(len(keys))
            table.setVerticalHeaderLabels([p['name'] for p in self.session.poses])
            table.setHorizontalHeaderLabels([f"{configs[k]['backend']} / {configs[k]['image_size']} / {configs[k]['mode']}\n{k[0][:8]} / {k[1]}" for k in keys])
            for row in range(len(self.session.poses)):
                for col,key in enumerate(keys):
                    result=self.session.latest(row,*key);value=None
                    if result:value=result.get('inference_ms') if metric.currentData()=='inference_ms' else result.get('metrics',{}).get(metric.currentData())
                    label=f'{value:.3f}' if value is not None else ('未评估' if result else '未完成/输入已变')
                    if result is None:
                        for run in reversed(self.session.data['runs']):
                            if (run['config_id'],run['selection'])!=key:continue
                            record=next((r for r in run['items'] if r['pose']==row+1),None)
                            if record and record['status']=='failed':label='失败：'+record.get('message','');break
                    item=QtWidgets.QTableWidgetItem(label)
                    item.setData(QtCore.Qt.UserRole,result);item.setToolTip(json.dumps(configs[key],ensure_ascii=False,indent=2));table.setItem(row,col,item)
            table.resizeColumnsToContents();table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            details,summary=self.session.export_csv(self.session.directory)
            selected=[r for r in summary if r['metric']==dict(inference_ms='inference_ms',mae_mm='depth_mae_mm',point_surface_mae_mm='point_surface_mae_mm',silhouette_iou='silhouette_iou',model_coverage_fraction='coverage')[metric.currentData()]]
            note.setText('共同有效姿态汇总（等权）：\n'+'\n'.join(f"{r['config_id'][:8]} / {r['selection']}：{r['common_pose_count']}/{r['total_pose_count']} 张；均值 {r['mean']:.3f}，中位数 {r['median']:.3f}，最差 {r['worst']:.3f}" if r['mean'] is not None else f"{r['config_id'][:8]}：无共同有效样本" for r in selected)+'\n点击单元格载入三维场景，关闭此窗口后查看。')
        def choose(row,col):
            item=table.item(row,col);result=item.data(QtCore.Qt.UserRole) if item else None
            if not result:return
            old=self.v.sample;view=dict(self.v.scene.opts)
            self.poses.setCurrentIndex(row);self.select(row)
            for i,r in enumerate(self.v.results):
                if r['label']==result['label']:self.v.table.selectRow(i);self.v.select_result();break
            if old==self.v.sample:self.v.scene.opts.update(view);self.v.scene.update()
        metric.currentIndexChanged.connect(populate);table.cellClicked.connect(choose);populate();dialog.exec_()
    def export_set(self):
        if not self.session or not self.idle():return
        parent=QtWidgets.QFileDialog.getExistingDirectory(self,'选择整组实验包导出位置')
        if not parent:return
        try:
            target=Path(parent)/('pose-set-'+self.session.directory.name)
            if target.resolve().is_relative_to(self.session.directory):raise ValueError('不能导出到源目录内部')
            self.session.export_csv(self.session.directory);shutil.copytree(self.session.directory,target)
            self.message('整组实验包和 CSV 已导出：'+str(target))
        except Exception as exc:self.message('导出失败：'+str(exc))
    def shutdown(self):
        self.timer.stop()
        if self.active and self.job['kind']=='reconstruct':self.job['run']['status']='interrupted';self.session.save()
        self.job=None
