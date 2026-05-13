"""Optional pynvml-backed GPU stats. Returns None on any error / missing dep."""
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
        h = _HANDLE_CACHE.get(device_index)
        if h is None:
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            _HANDLE_CACHE[device_index] = h
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        try:
            plim = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
        except Exception:
            plim = 0.0
        return GpuStats(name=name, mem_used_gb=mem.used / 1024**3, mem_total_gb=mem.total / 1024**3,
                        util_pct=int(util), temp_c=int(temp), power_w=pw, power_limit_w=plim)
    except Exception:
        return None
