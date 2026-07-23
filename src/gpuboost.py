"""Pin the GPU at full clocks for the duration of a training run (WSL -> Windows).

Why: with the console locked (e.g. an RDP session), the NVIDIA driver treats the
card as display-idle and steps it down over hours -- measured epochs went
636s -> 760s -> 956s at half clock (1500 MHz, cool, 96% CUDA load). WSL compute
never pulls the clocks back up on its own. Locking the clock floor to the card's
natural full-load boost (2700 MHz, ~340 W, 67 C on the 4090 -- well inside its
450 W / 84 C limits) restores and holds full speed.

Setting clocks needs Windows admin, which WSL doesn't have -- but WSL can run
schtasks.exe, and running your OWN highest-privilege scheduled task needs no UAC
prompt. So: two tasks are created ONCE from an admin PowerShell on Windows:

  schtasks /Create /TN "CV3GPUBoost"   /TR "nvidia-smi -lgc 2700,3135" /SC ONCE /ST 00:00 /RL HIGHEST /F
  schtasks /Create /TN "CV3GPURestore" /TR "nvidia-smi -rgc"           /SC ONCE /ST 00:00 /RL HIGHEST /F

(The ONCE schedule never fires on its own; the tasks exist purely to be /run on
demand with elevation.) After that, every training run pins the clocks at start
and restores driver defaults at the end automatically. If a run is hard-killed
(SIGKILL) the restore is skipped and the card idles hot until the next run ends,
`schtasks /run /tn CV3GPURestore`, or a reboot -- clock locks don't survive reboots.
"""
import os
import subprocess

SCHTASKS = "/mnt/c/Windows/System32/schtasks.exe"
BOOST_TASK = "CV3GPUBoost"
RESTORE_TASK = "CV3GPURestore"


def _run_task(name):
    if not os.path.exists(SCHTASKS):
        return False          # not on WSL (or no Windows side) -- silently no-op
    try:
        r = subprocess.run([SCHTASKS, "/run", "/tn", name],
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def boost():
    """Pin GPU clocks to full speed. Safe no-op when the task isn't set up."""
    if _run_task(BOOST_TASK):
        print("[GPU] Clocks pinned to full speed for this run (CV3GPUBoost).")
        return True
    print("[GPU] No clock pin (CV3GPUBoost task not set up) -- training runs at "
          "driver-managed clocks; see src/gpuboost.py for the one-time setup.")
    return False


def restore():
    """Give the driver its default clock management back (idle power savings)."""
    if _run_task(RESTORE_TASK):
        print("[GPU] Clocks restored to driver defaults (CV3GPURestore).")
        return True
    return False
