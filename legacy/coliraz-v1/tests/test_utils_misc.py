import time

from coliraz.utils.gpu import GpuStats, gpu_stats
from coliraz.utils.timing import EMA, Stopwatch


def test_gpu_stats_returns_none_or_dataclass():
    s = gpu_stats(device_index=0)
    assert s is None or isinstance(s, GpuStats)


def test_ema_smooths_values():
    ema = EMA(alpha=0.5)
    assert ema.update(10.0) == 10.0
    assert ema.update(20.0) == 15.0
    assert ema.update(20.0) == 17.5


def test_stopwatch_measures_time():
    sw = Stopwatch()
    sw.start()
    time.sleep(0.01)
    sw.stop()
    assert sw.elapsed > 0.005
