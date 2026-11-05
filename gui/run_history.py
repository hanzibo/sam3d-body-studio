# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Local run history with CSV export and immutable report links."""
import csv
import json
from pathlib import Path
from PyQt5 import QtWidgets
from launch_config import ROOT, LaunchConfig

HISTORY=ROOT/'outputs/gui/run-history.json'

def read_history():
    try:return json.loads(HISTORY.read_text())
    except (OSError,ValueError):return []

def record_report(directory,report):
    rows=read_history()
    bench=report.get('benchmark') or {}
    row=dict(directory=str(directory),config=report['actual_config'],config_id=report['config_id'],
             image_sha256=report['image_sha256'],image=report['image'],
             camera_intrinsics=report.get('camera_intrinsics_used'),
             camera_origin=report.get('camera_intrinsics_origin'),
             mean_ms=bench.get('mean_ms',report['inference_seconds']*1000),
             median_ms=bench.get('median_ms'),p95_ms=bench.get('p95_ms'),runs=bench.get('completed_runs',1),
             gui_seconds=report['gui_mesh_loaded_seconds'],kind='测速' if bench else '单次重建')
    rows.append(row);HISTORY.parent.mkdir(parents=True,exist_ok=True)
    temp=HISTORY.with_suffix('.tmp');temp.write_text(json.dumps(rows,ensure_ascii=False,indent=2));temp.replace(HISTORY)

def show_history(parent):
    rows=list(reversed(read_history()))
    dialog=QtWidgets.QDialog(parent);dialog.setWindowTitle('运行历史 · 同一图片及人物框才适合对照');dialog.resize(1150,500)
    layout=QtWidgets.QVBoxLayout(dialog)
    label=QtWidgets.QLabel('双击查看该次完整报告。单次重建只列平均栏；正式测速才提供中位数和 P95。')
    label.setWordWrap(True);layout.addWidget(label)
    columns=['结果目录','方案','图片哈希','内参','类型','次数','平均 ms','中位数 ms','P95 ms']
    table=QtWidgets.QTableWidget(len(rows),len(columns));table.setHorizontalHeaderLabels(columns)
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    values=[]
    for index,row in enumerate(rows):
        config=LaunchConfig.from_dict(row['config'])
        camera=row.get('camera_intrinsics')
        camera_label='未记录'
        if camera:
            camera_label=('真实相机' if camera['source']=='provided_rgb' else '默认估计')+f" · fx {camera['matrix'][0][0]:.1f}"
        data=[Path(row['directory']).name,config.summary(),row['image_sha256'][:12],camera_label,row['kind'],row['runs'],
              round(row['mean_ms'],2),round(row['median_ms'],2) if row['median_ms'] is not None else '',
              round(row['p95_ms'],2) if row['p95_ms'] is not None else '']
        values.append(data)
        for column,value in enumerate(data):table.setItem(index,column,QtWidgets.QTableWidgetItem(str(value)))
    table.resizeColumnsToContents();layout.addWidget(table)
    def details(index,column):
        path=Path(rows[index]['directory'])/'inference.json'
        popup=QtWidgets.QDialog(dialog);popup.setWindowTitle(str(path));popup.resize(900,600)
        box=QtWidgets.QVBoxLayout(popup);text=QtWidgets.QPlainTextEdit();text.setReadOnly(True)
        try:text.setPlainText(path.read_text())
        except OSError as exc:text.setPlainText(str(exc))
        box.addWidget(text);popup.exec_()
    table.cellDoubleClicked.connect(details)
    button=QtWidgets.QPushButton('导出 CSV')
    def export():
        path,_=QtWidgets.QFileDialog.getSaveFileName(dialog,'导出历史','studio-runs.csv','CSV (*.csv)')
        if path:
            with open(path,'w',newline='',encoding='utf-8-sig') as file:
                writer=csv.writer(file);writer.writerow(columns+['内参完整信息','内参来源','config_id','图片完整哈希','报告目录'])
                for data,row in zip(values,rows):
                    writer.writerow(data+[json.dumps(row.get('camera_intrinsics'),ensure_ascii=False),
                                          json.dumps(row.get('camera_origin'),ensure_ascii=False),
                                          row['config_id'],row['image_sha256'],row['directory']])
    button.clicked.connect(export);layout.addWidget(button);dialog.exec_()
