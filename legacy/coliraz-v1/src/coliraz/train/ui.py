"""Rich live dashboard for training."""
from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from coliraz.utils.gpu import gpu_stats
from coliraz.utils.timing import EMA


@dataclass
class _LossTrack:
    ema: EMA = field(default_factory=lambda: EMA(alpha=0.1))
    long_ema: EMA = field(default_factory=lambda: EMA(alpha=0.01))


class TrainUI(AbstractContextManager):
    def __init__(
        self, *, run_name: str, total_steps: int, headless: bool = False
    ) -> None:
        self.run_name = run_name
        self.total_steps = total_steps
        self.headless = headless
        self.console = Console()
        self._losses: dict[str, _LossTrack] = {}
        self._lr: float = 0.0
        self._throughput_imgs: float = 0.0
        self._step: int = 0
        self._last_preview: str = ""
        self._t0 = time.perf_counter()
        self._live: Live | None = None
        self._progress = Progress(
            TextColumn("step {task.completed}/{task.total}"),
            BarColumn(),
            TextColumn("{task.percentage:>5.1f}%"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self._task_id = self._progress.add_task("train", total=total_steps)

    def __enter__(self) -> "TrainUI":
        if not self.headless:
            self._live = Live(
                self.render(), refresh_per_second=6, console=self.console
            )
            self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._live:
            self._live.__exit__(*exc)
            self._live = None

    def tick(
        self, *, step: int, losses: dict[str, float], lr: float, throughput_imgs: float
    ) -> None:
        self._step = step
        self._lr = lr
        self._throughput_imgs = throughput_imgs
        for k, v in losses.items():
            t = self._losses.setdefault(k, _LossTrack())
            t.ema.update(v)
            t.long_ema.update(v)
        self._progress.update(self._task_id, completed=step)
        if self._live:
            self._live.update(self.render())

    def note_preview(self, msg: str) -> None:
        self._last_preview = msg
        if self._live:
            self._live.update(self.render())

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(Panel.fit(f"run: {self.run_name}", title="coliraz train"), size=3),
            Layout(self._progress, size=3),
            Layout(name="middle", size=14),
            Layout(
                Panel.fit(self._last_preview or "(no preview yet)", title="last preview"),
                size=3,
            ),
        )
        layout["middle"].split_row(self._losses_panel(), self._gpu_panel())
        return layout

    def _losses_panel(self) -> Panel:
        t = Table.grid(padding=(0, 1))
        t.add_column("loss")
        t.add_column("value", justify="right")
        t.add_column("trend", justify="right")
        for name, tr in self._losses.items():
            ema = tr.ema.value or 0.0
            longer = tr.long_ema.value or ema
            arrow = "▼" if ema < longer else "▲"
            t.add_row(name, f"{ema:.4f}", f"{arrow} {abs(ema - longer):.4f}")
        t.add_row("lr", f"{self._lr:.2e}", "")
        t.add_row("img/s", f"{self._throughput_imgs:.1f}", "")
        return Panel(t, title="losses (EMA)")

    def _gpu_panel(self) -> Panel:
        s = gpu_stats(0)
        if s is None:
            return Panel("gpu stats unavailable", title="gpu")
        t = Table.grid(padding=(0, 1))
        t.add_column()
        t.add_column(justify="right")
        t.add_row("name", s.name)
        t.add_row("mem", f"{s.mem_used_gb:.1f}/{s.mem_total_gb:.1f} GB")
        t.add_row("util", f"{s.util_pct}%")
        t.add_row("temp", f"{s.temp_c}°C")
        t.add_row("pwr", f"{s.power_w:.0f}/{s.power_limit_w:.0f} W")
        return Panel(t, title="gpu")
