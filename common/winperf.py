"""Keep long GPU jobs on fast CPU cores on Windows hybrid CPUs (Intel P-cores + E-cores).

Why: generation is bound by one Python thread that launches GPU kernels. When Windows treats the process as
background work (e.g. started from an editor terminal), it can run that thread on an efficiency core: the GPU
then sits ~5-10% busy and generation drops from ~12-15 to ~3 tokens/s (seen 2026-10-21; pinning the running
eval to the P-cores restored full speed within one turn).

boost_process() opts this process out of power throttling (EcoQoS), pins it to the performance cores, and asks
Windows not to idle-sleep while the process runs (SetThreadExecutionState). All are per-process, need no admin rights, and are skipped on other OSes, on CPUs without E-cores, or when
the environment variable AF_NO_CPU_BOOST=1 is set.
"""

from __future__ import annotations

import ctypes
import os
import sys

_PROCESS_POWER_THROTTLING = 4            # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_THROTTLE_EXECUTION_SPEED = 0x1
_RELATION_PROCESSOR_CORE = 0
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def performance_core_mask() -> int | None:
    """Affinity mask (processor group 0) of the fastest core class, or None if all cores are the same class."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformationEx(_RELATION_PROCESSOR_CORE, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetLogicalProcessorInformationEx(_RELATION_PROCESSOR_CORE, buf, ctypes.byref(size)):
        return None
    raw = buf.raw
    cores: list[tuple[int, int]] = []    # (efficiency class, group-0 mask)
    off = 0
    while off < size.value:
        rec_size = int.from_bytes(raw[off + 4:off + 8], "little")
        eff_class = raw[off + 9]
        group_count = int.from_bytes(raw[off + 30:off + 32], "little")
        for g in range(group_count):
            base = off + 32 + g * 16                                  # GROUP_AFFINITY: Mask (8), Group (2), pad (6)
            mask = int.from_bytes(raw[base:base + 8], "little")
            group = int.from_bytes(raw[base + 8:base + 10], "little")
            if group == 0:
                cores.append((eff_class, mask))
        off += rec_size
    classes = {c for c, _ in cores}
    if len(classes) < 2:
        return None
    best = max(classes)
    mask = 0
    for c, m in cores:
        if c == best:
            mask |= m
    return mask or None


def boost_process(verbose: bool = True) -> str:
    """Disable power throttling and pin to performance cores. Returns a short description of what was done."""
    if sys.platform != "win32" or os.environ.get("AF_NO_CPU_BOOST") == "1":
        return "cpu boost: skipped"
    done = []
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        proc = ctypes.c_void_p(k32.GetCurrentProcess())
        state = _PowerThrottlingState(1, _THROTTLE_EXECUTION_SPEED, 0)  # control execution speed, throttling off
        if k32.SetProcessInformation(proc, _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)):
            done.append("power throttling off")
        mask = performance_core_mask()
        if mask:
            k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            if k32.SetProcessAffinityMask(proc, mask):
                done.append(f"pinned to {bin(mask).count('1')} performance-core threads (mask 0x{mask:X})")
        # Keep the PC from idle-sleeping (Modern Standby) while this process runs: the v2 training paused for
        # 66 min on 2026-10-22 when the laptop went to standby. Held by this (main) thread until the process exits;
        # closing the lid can still sleep, depending on the lid setting.
        k32.SetThreadExecutionState.restype = ctypes.c_uint
        if k32.SetThreadExecutionState(ctypes.c_uint(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)):
            done.append("sleep blocked while running")
    except (OSError, AttributeError) as exc:  # never block a run over this
        done.append(f"failed: {exc}")
    msg = "cpu boost: " + (", ".join(done) if done else "nothing changed")
    if verbose:
        print(msg, flush=True)
    return msg
