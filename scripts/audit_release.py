#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Check tracked publication scope. Reports filenames only, never secret contents.

This is a narrow release guard, not a comprehensive security scanner.
"""
import argparse
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES={'.pt','.pth','.ckpt','.safetensors','.engine','.plan','.onnx','.pkl','.zip','.npz','.npy','.ply','.obj','.log'}
FORBIDDEN_ROOTS={'outputs','backups','Log','logs','ComfyUI','detectron2','third_party','models','checkpoints','envs','camera-analysis','community-review','fast-analysis'}
PATTERNS=[re.compile(value) for value in [
    r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{30,}',
    r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
    r'/(?:home/[^\s/]+|run/media/[^\s/]+)/',
]]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged',action='store_true')
    args=parser.parse_args()
    files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')[:-1]
    issues=[];total=0
    for name in files:
        p=Path(name)
        data=(subprocess.check_output(['git','show',':'+name],cwd=ROOT) if args.staged else (ROOT/p).read_bytes())
        total+=len(data)
        if p.parts[0] in FORBIDDEN_ROOTS or p.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append((name,'private/runtime/model artifact'))
        if p.parts[0]=='config' and (p.name.startswith('local.') or p.name=='gemini336l_rgb.json'):
            issues.append((name,'machine configuration'))
        if p.parts[:2]==('assets','smpl') and p.name!='README.md':issues.append((name,'SMPL asset'))
        if len(data)>2*1024**2:issues.append((name,'file larger than 2 MiB; review manually'))
        for number,line in enumerate(data.decode('utf-8',errors='replace').splitlines(),1):
            if any(pattern.search(line) for pattern in PATTERNS):issues.append((name,f'sensitive pattern at line {number}'))
    for name,reason in issues:print(name+': '+reason)
    print(f'Audited {len(files)} files, {total} bytes; {len(issues)} findings')
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
