import time

from restora_models.utils.gpu import GpuStats, gpu_stats
from restora_models.utils.timing import EMA, Stopwatch


def test_gpu_stats_returns_none_or_dataclass():
    s = gpu_stats(0)
    assert s is None or isinstance(s, GpuStats)


def test_ema_smooths_values():
    e = EMA(alpha=0.5)
    assert e.update(10.0) == 10.0
    assert e.update(20.0) == 15.0
    assert e.update(20.0) == 17.5


def test_stopwatch_measures_time():
    sw = Stopwatch().start()
    time.sleep(0.01)
    assert sw.stop() > 0.005
