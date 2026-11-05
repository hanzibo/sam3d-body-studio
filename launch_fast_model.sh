#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${STUDIO_BOOTSTRAP_PYTHON:-python3}" "$TASK_DIR/studio_paths.py" --launch fast "$@"
