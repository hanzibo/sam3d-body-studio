# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Hash pinning for locally built TensorRT engines, which are GPU/version specific.

Registration is a user trust decision, not a malware scan. Only register an
engine that you built locally or obtained from a trusted source.
"""
import argparse
import hashlib
import json
from pathlib import Path
from studio_paths import path


def verify_engine(filename, digest):
    registry = path('engine_registry')
    entries = json.loads(registry.read_text()) if registry.is_file() else {}
    expected = entries.get(str(Path(filename).resolve()), {}).get('sha256')
    if expected is None:
        raise RuntimeError(f'Unregistered TensorRT engine: {filename}; run engine_registry.py --register for your trusted local build')
    if expected != digest:
        raise RuntimeError(f'TensorRT engine hash mismatch: {filename}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', type=Path, required=True, help='Trusted local engine file')
    args = parser.parse_args()
    filename = args.register.expanduser().resolve()
    with filename.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    registry = path('engine_registry')
    entries = json.loads(registry.read_text()) if registry.is_file() else {}
    entries[str(filename)] = {'sha256': digest, 'size': filename.stat().st_size}
    registry.parent.mkdir(parents=True, exist_ok=True)
    temp = registry.with_suffix('.tmp')
    temp.write_text(json.dumps(entries, indent=2)+'\n'); temp.replace(registry)
    print(f'Registered {filename.name}: {digest}')

if __name__ == '__main__': main()
