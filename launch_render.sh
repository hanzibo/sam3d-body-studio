#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${STUDIO_BOOTSTRAP_PYTHON:-python3}" "$TASK_DIR/studio_paths.py" --launch render "$@"
