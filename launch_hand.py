# SPDX-License-Identifier: GPL-3.0-only
"""Small dependency-free launcher for the isolated original hand branch."""
import os
import sys
from pathlib import Path
from studio_paths import path, environment

root = Path(__file__).resolve().parent
env = environment('fast')
env['PYTHONPATH'] = os.pathsep.join((str(path('official_repo')),str(root)))
os.execve(str(path('fast_python')),[str(path('fast_python')),'-u','-m','studio.hand.hand_worker',*sys.argv[1:]],env)
