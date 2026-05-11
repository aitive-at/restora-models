"""Optional pynvml-backed GPU stats. Returns None on any error / missing dep.

The pynvml import is provided by either the legacy `pynvml` package or
nvidia's modern `nvidia-ml-py` (which exposes the same module name).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GpuStats:
    name: str
    mem_used_gb: float
    mem_total_gb: float
    util_pct: int
    temp_c: int
    power_w: float
    power_limit_w: float


_HANDLE_CACHE: dict[int, object] = {}
_INITIALIZED = False


def _ensure_init() -> bool:
    global _INITIALIZED
    if _INITIALIZED:
        return True
    try:
        import pynvml

        pynvml.nvmlInit()
        _INITIALIZED = True
        return True
    except Exception:
        return False


def gpu_stats(device_index: int = 0) -> GpuStats | None:
    if not _ensure_init():
        return None
    try:
        import pynvml

        handle = _HANDLE_CACHE.get(device_index)
        if handle is None:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            _HANDLE_CACHE[device_index] = handle
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        try:
            plim = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        except Exception:
            plim = 0.0
        return GpuStats(
            name=name,
            mem_used_gb=mem.used / 1024**3,
            mem_total_gb=mem.total / 1024**3,
            util_pct=int(util),
            temp_c=int(temp),
            power_w=pw,
            power_limit_w=plim,
        )
    except Exception:
        return None
