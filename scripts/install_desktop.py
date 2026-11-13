#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Install a desktop menu entry for this checkout, optionally on the desktop."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def quote(value):
    # Desktop Entry Exec quoting, not shell quoting. Percent signs are field codes.
    value=str(value).replace('%','%%')
    for character in ('\\','"','`','$'):
        value=value.replace(character,'\\'+character)
    return '"'+value+'"'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--desktop', action='store_true')
    parser.add_argument('--recorder', action='store_true', help='Install the independent RGB-D recorder instead')
    args=parser.parse_args()
    name='RGB-D 数据采集' if args.recorder else 'SAM 3D Body Studio'
    launcher='launch_rgbd_recorder.sh' if args.recorder else 'launch_gui.sh'
    identifier='sam3d-rgbd-recorder' if args.recorder else 'sam3d-body-studio'
    content=('''[Desktop Entry]
Type=Application
Comment=SAM 3D Body / Fast SAM 3D Body desktop experiments
Terminal=false
Categories=Graphics;Science;
'''+f'Name={name}\nExec={quote(ROOT/launcher)}\nIcon={ROOT/"gui/icon.svg"}\nPath={ROOT}\n')
    targets=[Path.home()/'.local/share/applications'/f'{identifier}.desktop']
    if args.desktop:
        result=subprocess.run(['xdg-user-dir','DESKTOP'], capture_output=True, text=True, check=True)
        targets.append(Path(result.stdout.strip())/('RGB-D-Recorder.desktop' if args.recorder else 'SAM-3D-Body.desktop'))
    for target in targets:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(content);target.chmod(0o755)
        print(target)
    print('If prompted by GNOME, choose Allow Launching on the desktop icon.')

if __name__=='__main__': main()
