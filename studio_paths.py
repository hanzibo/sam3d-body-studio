"""Machine-local paths and process environments; no model imports at GUI startup.

Copy config/paths.example.json to config/local.json, or set SAM3D_STUDIO_PATHS.
Relative paths resolve against the Studio checkout, never the caller's cwd.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = Path(os.environ.get('SAM3D_STUDIO_PATHS', ROOT/'config/local.json')).expanduser().resolve()
DEFAULTS = {
    'official_repo': 'third_party/sam-3d-body',
    'fast_repo': 'third_party/Fast-SAM-3D-Body',
    'comfy_repo': 'third_party/ComfyUI',
    'sam3_repo': 'third_party/sam3',
    'standard_python': 'envs/standard/bin/python',
    'fast_python': 'envs/fast/bin/python',
    'camera_python': 'envs/camera/bin/python',
    'sam3_python': 'envs/sam3/bin/python',
    'standard_checkpoint': 'models/sam_3d_body_dinov3_bf16.safetensors',
    'sam3_checkpoint': 'models/sam3/sam3.pt',
    'backbone_engine_512': 'models/engines/512/backbone_dinov3_fp16.engine',
    'backbone_engine_384': 'models/engines/384/backbone_dinov3_fp16.engine',
    'torch_home': 'models/torch-hub',
    'inductor_cache': 'models/torch-cache/fast',
    'yolo_config': 'outputs/yolo-config',
    'smpl_assets': 'assets/smpl',
    'engine_registry': 'config/local.engines.json',
}
DERIVED = {
    'fast_checkpoint_dir': ('fast_repo', 'checkpoints/sam-3d-body-dinov3'),
    'yolo_engine': ('fast_repo', 'checkpoints/yolo/yolo11m-pose.engine'),
    'yolo_checkpoint': ('fast_repo', 'checkpoints/yolo/yolo11m-pose.pt'),
    'dinov3_repo': ('torch_home', 'hub/facebookresearch_dinov3_main'),
}

def read_config():
    value = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.is_file() else {}
    if not isinstance(value, dict):
        raise ValueError('Path configuration must be a JSON object')
    unknown = value.keys() - (DEFAULTS.keys() | DERIVED.keys())
    if unknown:
        raise ValueError('Unknown path configuration keys: '+', '.join(sorted(unknown)))
    if any(not isinstance(v, str) or not v.strip() for v in value.values()):
        raise ValueError('All configured paths must be non-empty strings')
    return value

VALUES = read_config()

def path(key):
    if key in VALUES:
        result = Path(os.path.expandvars(VALUES[key])).expanduser()
    elif key in DERIVED:
        parent, suffix = DERIVED[key]
        return path(parent)/suffix
    else:
        result = Path(DEFAULTS[key])
    return result.resolve() if result.is_absolute() else (ROOT/result).resolve()


def environment(kind):
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYOPENGL_PLATFORM', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    if kind == 'gui':
        env.update(QT_API='pyqt5', QT_QPA_PLATFORM=env.get('QT_QPA_PLATFORM', 'xcb'))
    elif kind != 'camera':
        source = [path('fast_repo')] if kind in ('fast', 'build') else [path('comfy_repo'), path('official_repo')]
        env.update(PYTHONPATH=os.pathsep.join(map(str, source)), PYOPENGL_PLATFORM='egl',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='2',
                   TORCH_HOME=str(path('torch_home')), TORCHINDUCTOR_CACHE_DIR=str(path('inductor_cache')),
                   YOLO_CONFIG_DIR=str(path('yolo_config')))
    return env


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', choices=['gui','standard','fast','camera','render','single','build'])
    parser.add_argument('--doctor', action='store_true')
    args, extra = parser.parse_known_args()
    if args.doctor:
        if extra: parser.error('Unexpected doctor arguments')
        items = {k: {'path': str(path(k)), 'exists': path(k).exists()} for k in [*DEFAULTS, *DERIVED]}
        print(json.dumps({'config': str(CONFIG_FILE), 'configured': CONFIG_FILE.is_file(), 'paths': items}, indent=2))
        return
    if not args.launch: parser.error('Specify --launch or --doctor')
    kind = args.launch
    interpreter = {'gui':'standard','standard':'standard','fast':'fast','camera':'camera',
                   'render':'standard','single':'standard','build':'fast'}[kind]
    executable = path(interpreter+'_python')
    if not executable.is_file():
        parser.exit(1, f'Python environment missing: {executable}\nConfigure {CONFIG_FILE}; see docs/INSTALL.md\n')
    script = {'gui':'gui/app.py','standard':'resident_worker.py','fast':'fast_worker.py',
              'camera':'camera_worker.py','render':'render_comparison.py','single':'run_comfy.py',
              'build':'build_backbone.py'}[kind]
    if kind == 'gui':
        folder = ROOT/'outputs/gui'; folder.mkdir(parents=True, exist_ok=True)
        with (folder/'desktop.log').open('ab') as log:
            os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    os.chdir(ROOT)
    os.execve(str(executable), [str(executable), '-u', str(ROOT/script), *extra], environment(kind))

if __name__ == '__main__': main()
