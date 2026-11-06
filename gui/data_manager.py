# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Artifact browser: background inventory, bounded previews, recoverable deletion."""
from datetime import datetime
from pathlib import Path
import shutil

from PyQt5 import QtCore, QtGui, QtWidgets
from data_catalog import scan, checked_path, validate_delete, human_size

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.json', '.jsonl', '.csv', '.txt', '.log', '.yaml', '.yml', '.md'}


class TrashProgress(QtWidgets.QProgressDialog):
    """Keep user-triggered new jobs blocked until the filesystem operation ends."""
    allow_close = False

    def reject(self):
        if self.allow_close: super().reject()

    def closeEvent(self, event):
        if self.allow_close: super().closeEvent(event)
        else: event.ignore()


class DataTask(QtCore.QThread):
    completed = QtCore.pyqtSignal(object, str)

    def __init__(self, root, entries=None, protected=(), parent=None):
        super().__init__(parent)
        self.root, self.entries, self.protected = root, entries, protected

    def run(self):
        try:
            if self.entries is None:
                self.completed.emit(scan(self.root, self.isInterruptionRequested), '')
            else:
                removed, errors = [], []
                for entry in self.entries:
                    try:
                        path = validate_delete(self.root, entry, self.protected)
                        ok, trash_path = QtCore.QFile.moveToTrash(str(path))
                        if not ok:
                            raise OSError('系统回收站不可用或权限不足；原数据保留，不回退为永久删除')
                        removed.append(str(path))
                    except (ValueError, OSError) as exc:
                        errors.append(f'{entry.path.name}: {exc}')
                self.completed.emit(removed, '\n'.join(errors))
        except Exception as exc:
            self.completed.emit([], str(exc))


class DataManager(QtWidgets.QWidget):
    def __init__(self, host, root=ROOT):
        super().__init__()
        self.host, self.root = host, Path(root).resolve()
        self.entries = []; self.task = None; self.progress = None; self.scanned = False
        self.deleting = False
        self.stopping = False
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel('按完整样本 / 整次运行管理，不拆散 RGB-D、分割、预测与评分。删除移入系统回收站；清空回收站后才释放空间。')
        note.setWordWrap(True); layout.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        self.category = QtWidgets.QComboBox(); self.category.addItem('全部类别')
        self.category.setMinimumWidth(205)
        self.category.currentTextChanged.connect(self.filter_rows)
        self.order = QtWidgets.QComboBox(); self.order.addItems(['最近修改优先', '占用空间：从大到小', '占用空间：从小到大', '类别 / 名称'])
        self.order.setMinimumWidth(175)
        self.order.setToolTip('按实际字节数排序；也可点击“大小（逻辑）”表头切换升序 / 降序。')
        self.order.currentIndexChanged.connect(lambda: self.scanned_entries(self.entries, '') if self.scanned and self.task is None else None)
        self.search = QtWidgets.QLineEdit(); self.search.setPlaceholderText('搜索名称、路径或类别'); self.search.textChanged.connect(self.filter_rows)
        self.refresh_btn = QtWidgets.QPushButton('刷新 / 重新统计'); self.refresh_btn.clicked.connect(self.refresh)
        self.open_btn = QtWidgets.QPushButton('打开所在目录'); self.open_btn.clicked.connect(self.open_folder)
        self.delete_btn = QtWidgets.QPushButton('移入回收站…'); self.delete_btn.clicked.connect(self.delete_selected)
        for widget in (self.category, self.order, self.search, self.refresh_btn, self.open_btn, self.delete_btn): row.addWidget(widget)
        layout.addLayout(row)
        self.summary = QtWidgets.QLabel('切换到本页后自动统计；模型、引擎、环境、源码和本机配置不在清理范围。')
        self.summary.setWordWrap(True); layout.addWidget(self.summary)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical); layout.addWidget(split, 1)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['类别', '数据名称', '大小（逻辑）', '文件数', '最近修改', '相对路径'])
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionClicked.connect(self.sort_column)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        self.table.cellDoubleClicked.connect(lambda *_: self.open_folder())
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu); self.table.customContextMenuRequested.connect(self.context_menu)
        split.addWidget(self.table)
        detail = QtWidgets.QSplitter(); split.addWidget(detail)
        self.files = QtWidgets.QTreeWidget(); self.files.setHeaderLabels(['文件 / 子目录', '大小'])
        self.files.itemExpanded.connect(self.expand_folder); self.files.currentItemChanged.connect(self.preview_file)
        detail.addWidget(self.files)
        pane = QtWidgets.QWidget(); pl = QtWidgets.QVBoxLayout(pane)
        self.info = QtWidgets.QLabel('选择一条数据查看目录和说明；可多选移入回收站。'); self.info.setWordWrap(True); pl.addWidget(self.info)
        self.image = QtWidgets.QLabel(); self.image.setAlignment(QtCore.Qt.AlignCenter); self.image.setMinimumHeight(180); pl.addWidget(self.image)
        self.image.hide()
        self.text = QtWidgets.QPlainTextEdit(); self.text.setReadOnly(True); pl.addWidget(self.text, 1)
        self.text.setPlaceholderText('展开左侧目录并点击图片、配置、报告或日志进行预览。')
        detail.addWidget(pane); detail.setStretchFactor(1, 2)
        self.status = QtWidgets.QLabel(); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.guard = QtWidgets.QLabel(); self.guard.setWordWrap(True); layout.addWidget(self.guard)
        self.timer = QtCore.QTimer(self); self.timer.setInterval(500); self.timer.timeout.connect(self.update_controls); self.timer.start()
        self.update_controls()

    def showEvent(self, event):
        super().showEvent(event)
        if self.task is None: self.refresh()

    def busy_reason(self):
        h = self.host; v = h.rgbd; lab = h.lab.client
        if hasattr(h,'video') and h.video.active:return '视频任务正在运行'
        if hasattr(h,'image_seg') and h.image_seg.active: return '第一页采集 / SAM3 分割正在运行'
        if h.inference_busy or h.model_state in ('loading', 'stopping') or h.pending_launch:
            return '模型正在加载、推理或释放'
        if h.camera.active or h.camera_state == 'countdown' or (getattr(h, 'live', None) and h.live.running):
            return '相机 / 实时流正在运行，请先停止'
        if v.mode is not None or v.process.state() != QtCore.QProcess.NotRunning or v.expected is not None or v.segmentation_pending is not None or v.pending_camera_action is not None or v.batch.active:
            return 'RGB-D 采集、分割或批处理正在运行'
        if any(p.state() != QtCore.QProcess.NotRunning for p in h.render_processes.values()):
            return '对照图正在后台渲染'
        if h.motion.recorder is not None:
            return '动作录制仍在写入，请先停止并等待完成'
        if lab.inflight is not None or lab.pending is not None or h.lab.timer.isActive():
            return 'MHR 实验室正在计算'
        return ''

    def protected_paths(self):
        h = self.host; v = h.rgbd
        paths = [h.source_path, h.result_dir, v.sample,
                 h.settings.value('lab_latest_prediction_path', '')]
        if hasattr(h,'image_seg') and h.image_seg.directory: paths.append(h.image_seg.directory)
        if hasattr(h,'video') and h.video.directory: paths.append(h.video.directory)
        if h.model_state != 'off' and h.model_log_path:
            paths.extend([h.model_log_path, Path(h.model_log_path).with_suffix('.json')])
        if Path(h.lab.source.text()).is_absolute(): paths.append(h.lab.source.text())
        if v.batch.session: paths.append(v.batch.session.directory)
        if h.lab.latest: paths.append(h.lab.latest[1])
        # A completed run's working directory no longer receives writes.
        if h.inference_busy: paths.append(h.running_dir)
        return [p for p in paths if p and isinstance(p, (str, Path))]

    def selected_entries(self):
        return [self.entries[i.row()] for i in self.table.selectionModel().selectedRows() if not self.table.isRowHidden(i.row())]

    def update_controls(self):
        entries = self.selected_entries()
        self.refresh_btn.setEnabled(self.task is None)
        self.order.setEnabled(self.task is None)
        self.open_btn.setEnabled(bool(entries) and not self.deleting)
        reason = self.busy_reason()
        if not reason:
            try:
                for entry in entries: validate_delete(self.root, entry, self.protected_paths())
            except (ValueError, OSError) as exc: reason = str(exc)
        self.delete_btn.setEnabled(bool(entries) and self.task is None and not reason)
        self.delete_btn.setToolTip(reason or '将所选完整数据移入系统回收站，可从文件管理器恢复。')
        self.guard.setText('暂不可删除：' + reason if reason else '')

    def refresh(self):
        if self.task is not None or self.stopping: return
        self.status.setText('后台统计中…大小为文件逻辑字节数，硬链接/稀疏文件占用可能不同。')
        self.task = DataTask(self.root, parent=self)
        self.task.completed.connect(self.scanned_entries)
        self.task.finished.connect(self.task_finished)
        self.task.start(); self.update_controls()

    def scanned_entries(self, entries, error):
        order = self.order.currentIndex()
        keys = [lambda e: -e.modified, lambda e: -e.size, lambda e: e.size, lambda e: (e.category, e.path.name)]
        entries = sorted(entries, key=keys[order])
        header = self.table.horizontalHeader()
        header.setSortIndicatorShown(order in (1, 2))
        if order in (1, 2): header.setSortIndicator(2, QtCore.Qt.DescendingOrder if order == 1 else QtCore.Qt.AscendingOrder)
        self.entries = entries; self.scanned = True
        selected_category = self.category.currentText()
        self.category.blockSignals(True); self.category.clear(); self.category.addItem('全部类别')
        self.category.addItems(sorted({e.category for e in entries}))
        index = self.category.findText(selected_category); self.category.setCurrentIndex(max(index, 0)); self.category.blockSignals(False)
        self.table.setRowCount(0); self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            values = [entry.category, entry.path.name, human_size(entry.size), str(entry.files),
                      datetime.fromtimestamp(entry.modified).strftime('%Y-%m-%d %H:%M:%S') if entry.modified else '—',
                      str(entry.path.relative_to(self.root))]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(entry.description + ('\n' + entry.warning if entry.warning else '') + ('\n只读' if not entry.deletable else ''))
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents(); self.filter_rows()
        disk = shutil.disk_usage(self.root)
        self.summary.setText(f'应用目录：{self.root}\n已登记 {len(entries)} 项 · {human_size(sum(e.size for e in entries))} · 所在磁盘可用 {human_size(disk.free)} / {human_size(disk.total)}')
        self.status.setText('扫描失败：' + error if error else '统计完成。只读项和正在使用的数据不能删除。手动导出到其他目录的文件不在此清单。')
        self.files.clear(); self.image.clear(); self.image.hide(); self.text.clear()

    def task_finished(self):
        task = self.task; self.task = None
        if task: task.deleteLater()
        self.update_controls()

    def sort_column(self, column):
        if column == 2 and self.task is None:
            self.order.setCurrentIndex(2 if self.order.currentIndex() == 1 else 1)

    def filter_rows(self, *_):
        query = self.search.text().strip().lower(); category = self.category.currentText()
        for row, e in enumerate(self.entries):
            hidden = (category != '全部类别' and e.category != category) or query not in f'{e.path} {e.category}'.lower()
            self.table.setRowHidden(row, hidden)
        self.update_controls()

    def selection_changed(self):
        self.files.clear(); self.image.clear(); self.image.hide(); self.text.clear()
        entries = self.selected_entries()
        if entries:
            entry = entries[0]
            self.info.setText(f'{entry.path}\n{entry.description}\n{entry.warning}\n所选 {len(entries)} 项，共 {human_size(sum(e.size for e in entries))}')
            try:
                p = checked_path(self.root, entry.path)
                if p.is_dir(): self.add_children(self.files.invisibleRootItem(), p)
                else: self.preview_path(p)
            except (ValueError, OSError) as exc: self.text.setPlainText(str(exc))
        self.update_controls()

    def add_children(self, parent, folder):
        children = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        for path in children[:2000]:
            info = path.lstat()
            item = QtWidgets.QTreeWidgetItem([path.name, '链接（禁用）' if path.is_symlink() else ('目录' if path.is_dir() else human_size(info.st_size))])
            item.setData(0, QtCore.Qt.UserRole, str(path)); parent.addChild(item)
            if path.is_dir() and not path.is_symlink(): item.setChildIndicatorPolicy(QtWidgets.QTreeWidgetItem.ShowIndicator)
        if len(children) > 2000: parent.addChild(QtWidgets.QTreeWidgetItem(['仅展示前 2000 项，请打开所在目录查看全部']))

    def expand_folder(self, item):
        if item.childCount(): return
        try: self.add_children(item, checked_path(self.root, item.data(0, QtCore.Qt.UserRole)))
        except (ValueError, OSError) as exc: self.status.setText(str(exc))

    def preview_file(self, item, *_):
        if item and item.data(0, QtCore.Qt.UserRole): self.preview_path(Path(item.data(0, QtCore.Qt.UserRole)))

    def preview_path(self, file):
        self.image.clear(); self.image.hide(); self.text.clear()
        try:
            file = checked_path(self.root, file)
            if not file.is_file(): return
            size = file.stat().st_size
            if file.suffix.lower() in TEXT_SUFFIXES:
                with file.open('rb') as stream: content = stream.read(262144)
                self.text.setPlainText(content.decode('utf-8', errors='replace') + ('\n…仅预览前 256 KiB' if size > 262144 else ''))
            elif file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                reader = QtGui.QImageReader(str(file))
                dimensions = reader.size()
                if dimensions.width() * dimensions.height() > 40000000 or size > 32*1024*1024:
                    raise ValueError('图片过大，跳过内置预览，请打开所在目录查看')
                reader.setScaledSize(dimensions.scaled(640, 360, QtCore.Qt.KeepAspectRatio))
                image = reader.read()
                if image.isNull(): raise ValueError('无法解码图片')
                self.image.setPixmap(QtGui.QPixmap.fromImage(image))
                self.image.show()
            else:
                self.text.setPlainText(f'{file.name}\n{human_size(size)}\n二进制 / 网格文件仅列出，不反序列化或执行。')
        except (ValueError, OSError) as exc: self.text.setPlainText(str(exc))

    def open_folder(self):
        entries = self.selected_entries()
        if not entries: return
        try:
            p = checked_path(self.root, entries[0].path)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p if p.is_dir() else p.parent)))
        except (ValueError, OSError) as exc: self.status.setText(str(exc))

    def context_menu(self, position):
        index = self.table.indexAt(position)
        if not index.isValid(): return
        if not self.table.selectionModel().isRowSelected(index.row(), QtCore.QModelIndex()): self.table.selectRow(index.row())
        self.update_controls(); menu = QtWidgets.QMenu(self)
        menu.addAction('打开所在目录', self.open_folder)
        action = menu.addAction('移入回收站…', self.delete_selected); action.setEnabled(self.delete_btn.isEnabled())
        menu.exec_(self.table.viewport().mapToGlobal(position))

    def delete_selected(self):
        entries = self.selected_entries()
        if not entries or self.task is not None: return
        try:
            reason = self.busy_reason()
            if reason: raise ValueError(reason)
            for e in entries: validate_delete(self.root, e, self.protected_paths())
            names = '\n'.join(str(e.path.relative_to(self.root)) for e in entries[:20])
            if len(entries) > 20: names += f'\n…另 {len(entries)-20} 项'
            answer = QtWidgets.QMessageBox.question(self, '将完整数据移入回收站？',
                f'所选 {len(entries)} 项，共 {human_size(sum(e.size for e in entries))}：\n{names}\n\n'
                '目录内的照片、掩膜、网格、配置、评估和日志一并移走。\n历史索引保留，源图片 / 结果链接可能显示缺失；恢复回收站数据后可继续使用。\n'
                '实验室导出的同名 NPZ / JSON / PLY 为独立条目，请同时选中。\n不会永久删除；清空系统回收站后才释放空间。',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes: return
            # Modal confirmation runs an event loop; recheck job state afterwards.
            reason = self.busy_reason()
            if reason: raise ValueError(reason)
            protected = self.protected_paths()
            for e in entries: validate_delete(self.root, e, protected)
            self.deleting = True
            self.progress = TrashProgress('正在移入系统回收站…', '', 0, 0, self)
            self.progress.setCancelButton(None); self.progress.setWindowModality(QtCore.Qt.ApplicationModal)
            self.progress.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False); self.progress.show()
            self.task = DataTask(self.root, entries, protected, self)
            self.task.completed.connect(self.deleted_entries); self.task.finished.connect(self.deletion_finished)
            self.task.start(); self.update_controls()
        except (ValueError, OSError) as exc: self.status.setText('未删除：' + str(exc))

    def deleted_entries(self, removed, error):
        self.status.setText(f'已移入回收站 {len(removed)} 项。' + ('\n部分失败：' + error if error else '可在系统文件管理器的回收站恢复。'))
        if error: QtWidgets.QMessageBox.warning(self, '部分数据未删除', error)

    def deletion_finished(self):
        self.task_finished(); self.deleting = False
        if self.progress:
            self.progress.allow_close = True
            self.progress.close(); self.progress.deleteLater(); self.progress = None
        # Preserve the operation result below the refreshed inventory.
        result = self.status.text()
        if not self.stopping:
            self.refresh()
            self.task.finished.connect(lambda: self.status.setText(result))

    def shutdown(self):
        self.stopping = True
        self.timer.stop()
        if self.task:
            self.task.requestInterruption(); self.task.wait()
