#!/usr/bin/env python3
"""Fetch pinned source only; never download model weights or install packages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio_paths import path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam3', action='store_true', help='Also fetch optional SAM3 source')
    args = parser.parse_args()
    for item in json.loads((ROOT/'dependencies.json').read_text()):
        if item.get('optional') and not args.sam3: continue
        destination = ROOT/'third_party/detectron2' if item['key']=='detectron2' else path(item['key'])
        if destination.exists():
            actual = subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'], text=True).strip()
            if actual != item['commit']:
                raise RuntimeError(f'{destination} already exists at {actual}; expected {item["commit"]}. No changes made to this checkout.')
            print(f'Already pinned: {item["key"]}', flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',item['url'],str(destination)], check=True)
        subprocess.run(['git','-C',str(destination),'checkout','--detach',item['commit']], check=True)
        print(f'Fetched: {item["key"]}', flush=True)

if __name__=='__main__': main()
