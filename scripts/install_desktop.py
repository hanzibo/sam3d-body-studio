#!/usr/bin/env python3
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
    args=parser.parse_args()
    content=('''[Desktop Entry]
Type=Application
Name=SAM 3D Body Studio
Comment=SAM 3D Body / Fast SAM 3D Body desktop experiments
Terminal=false
Categories=Graphics;Science;
'''+f'Exec={quote(ROOT/"launch_gui.sh")}\nIcon={ROOT/"gui/icon.svg"}\nPath={ROOT}\n')
    targets=[Path.home()/'.local/share/applications/sam3d-body-studio.desktop']
    if args.desktop:
        result=subprocess.run(['xdg-user-dir','DESKTOP'], capture_output=True, text=True, check=True)
        targets.append(Path(result.stdout.strip())/'SAM-3D-Body.desktop')
    for target in targets:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(content);target.chmod(0o755)
        print(target)
    print('If prompted by GNOME, choose Allow Launching on the desktop icon.')

if __name__=='__main__': main()
