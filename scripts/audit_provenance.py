#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Check inventory coverage and license snapshots, not legal compatibility."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def audit(root=ROOT):
    inventory = json.loads((root / 'docs/provenance.json').read_text())
    tracked = set(subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=root).decode().split('\0')[:-1])
    entries = inventory['files']
    issues = []
    for name in sorted(tracked - entries.keys()):
        issues.append(f'Missing provenance entry: {name}')
    for name in sorted(entries.keys() - tracked):
        issues.append(f'Untracked provenance entry: {name}')
    for name, entry in entries.items():
        if not entry.get('basis') or not entry.get('license'):
            issues.append(f'Incomplete provenance entry: {name}')
        if name in tracked and Path(name).suffix in ('.py', '.sh'):
            expected = '# SPDX-License-Identifier: ' + entry['license']
            if expected not in (root / name).read_text().splitlines()[:12]:
                issues.append(f'License header does not match inventory: {name}')
    for name, entry in inventory['license_snapshots'].items():
        if name not in tracked:
            issues.append(f'Untracked license snapshot: {name}')
        elif hashlib.sha256((root / name).read_bytes()).hexdigest() != entry['sha256']:
            issues.append(f'License snapshot changed: {name}')
    return issues, len(entries)


def main():
    issues, count = audit()
    for issue in issues:
        print(issue)
    print(f'Provenance inventory: {count} files; {len(issues)} consistency findings')
    print('This check does not establish authorship, model authorization or license compatibility.')
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
