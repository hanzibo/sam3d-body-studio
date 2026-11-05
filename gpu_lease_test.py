"""Prove cross-process exclusion and automatic release after hard process death."""
import subprocess
import sys
from gpu_lease import acquire_gpu_lease

def holder(kind):
    p=subprocess.Popen([sys.executable,'-u','-c',f"from gpu_lease import acquire_gpu_lease; import time; lock=acquire_gpu_lease('{kind}'); print('ready',flush=True);time.sleep(60)"],stdout=subprocess.PIPE,text=True)
    assert p.stdout.readline().strip()=='ready'
    return p

def blocked(kind):
    try:l=acquire_gpu_lease(kind)
    except RuntimeError:return
    l.close();raise AssertionError(kind+' was not blocked')

p=holder('body')
try:
    blocked('sam3')
    # Existing body + small MHR laboratory behavior remains supported.
    lab=acquire_gpu_lease('mhr');lab.close()
finally:p.kill();p.wait()
p=holder('sam3')
try:
    blocked('body');blocked('mhr');blocked('sam3')
finally:p.kill();p.wait()
for kind in ('body','mhr','sam3'):
    lease=acquire_gpu_lease(kind);lease.close()
print('PASS body/MHR vs SAM3 exclusion, two SAM3 workers blocked, crash recovery')
