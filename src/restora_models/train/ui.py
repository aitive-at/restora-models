"""Rich live dashboard with per-task PSNR + LPIPS rows."""
from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from restora_models.utils.gpu import gpu_stats
from restora_models.utils.timing import EMA


def _fmt_hms(seconds: float) -> str:
    """`H:MM:SS` for a non-negative second count. Used for both elapsed
    and ETA displays — we keep hours unpadded so `12:34:56` and `1:23:45`
    both read naturally."""
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


@dataclass
class _EMATrack:
    short: EMA = field(default_factory=lambda: EMA(alpha=0.1))
    long: EMA = field(default_factory=lambda: EMA(alpha=0.01))


class _LiveLayoutFactory:
    """Re-render the dashboard on every ``rich.live.Live`` refresh tick
    (default 6 Hz), not just when ``ui.tick()`` fires. Live calls
    ``__rich_console__`` once per refresh; routing it back through
    ``ui.render()`` lets the elapsed/ETA timer and the GPU panel advance
    in real time even when ``log_every`` ticks are minutes apart at
    large batch sizes."""

    def __init__(self, ui: "TrainUI") -> None:
        self._ui = ui

    def __rich_console__(self, console, options):  # noqa: D401 - rich proto
        yield self._ui.render()


class TrainUI(AbstractContextManager):
    def __init__(self, *, run_name: str, total_steps: int, headless: bool = False,
                 task_names: list[str] | None = None) -> None:
        self.run_name = run_name
        self.total_steps = total_steps
        self.headless = headless
        self.task_names = task_names or []
        self.console = Console()
        self._losses: dict[str, _EMATrack] = {}
        self._psnr: dict[str, _EMATrack] = {n: _EMATrack() for n in self.task_names}
        self._lpips: dict[str, _EMATrack] = {n: _EMATrack() for n in self.task_names}
        self._lr = 0.0
        self._throughput = 0.0
        self._grad_norm = 0.0
        self._step = 0
        self._last_preview = ""
        self._t0 = time.perf_counter()
        self._live: Live | None = None
        # We compute elapsed / ETA ourselves rather than relying on rich's
        # TimeRemainingColumn — that one needs a rolling window of >=3
        # Progress.update() samples to estimate speed, but we only tick
        # every `log_every` steps so it stays blank for the first few
        # minutes of every run.
        self._progress = Progress(
            TextColumn("step {task.completed}/{task.total}"),
            BarColumn(),
            TextColumn("{task.percentage:>5.1f}%"),
            TextColumn("•"),
            TextColumn("[bold]{task.fields[elapsed_str]}[/bold] elapsed"),
            TextColumn("•"),
            TextColumn("[bold]{task.fields[eta_str]}[/bold] ETA"),
            console=self.console,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            "train", total=total_steps, elapsed_str="0:00:00", eta_str="--:--:--")

    def __enter__(self) -> "TrainUI":
        if not self.headless:
            # Pass a factory (not a static Layout) so each 6 Hz refresh
            # rebuilds the Layout — keeps elapsed/ETA/GPU panel live
            # between log_every ticks.
            self._live = Live(_LiveLayoutFactory(self),
                              refresh_per_second=6, console=self.console)
            self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._live:
            self._live.__exit__(*exc)
            self._live = None

    def tick(self, *, step: int, losses: dict[str, float], lr: float,
             throughput_imgs: float,
             per_task_psnr: dict[str, float] | None = None,
             per_task_lpips: dict[str, float] | None = None,
             grad_norm: float | None = None) -> None:
        self._step = step
        self._lr = lr
        self._throughput = throughput_imgs
        if grad_norm is not None:
            self._grad_norm = float(grad_norm)
        for k, v in losses.items():
            if not isinstance(v, (int, float)):
                continue
            t = self._losses.setdefault(k, _EMATrack())
            t.short.update(v); t.long.update(v)
        if per_task_psnr:
            for k, v in per_task_psnr.items():
                if v == v:  # NaN-guard
                    track = self._psnr.setdefault(k, _EMATrack())
                    track.short.update(v); track.long.update(v)
        if per_task_lpips:
            for k, v in per_task_lpips.items():
                if v == v:
                    track = self._lpips.setdefault(k, _EMATrack())
                    track.short.update(v); track.long.update(v)
        # In Live mode the 6 Hz refresh thread re-renders via
        # _LiveLayoutFactory → render(), which recomputes elapsed/ETA
        # and re-reads pynvml on every frame. We only need an explicit
        # headless print here, since headless has no refresh loop.
        if self.headless:
            elapsed_s = time.perf_counter() - self._t0
            eta_s = self._eta_seconds(elapsed_s, step)
            self._print_headless(elapsed_s=elapsed_s, eta_s=eta_s)

    def _eta_seconds(self, elapsed_s: float, step: int) -> float | None:
        # Wall-clock projection from total-average step rate. This biases
        # the estimate upward early on (compile warmup eats the first
        # tick) but self-corrects within a few minutes — and it never
        # goes blank, which is the whole point.
        if step <= 0 or self.total_steps <= 0 or step >= self.total_steps:
            return None
        return elapsed_s * (self.total_steps - step) / step

    def _print_headless(self, *, elapsed_s: float, eta_s: float | None) -> None:
        """One-line summary printed each tick in headless mode (no TTY).

        Picks the most informative scalar loss available (``total`` first,
        else ``loss``, else ``l1_rgb``) and appends each tracked per-axis
        PSNR + LPIPS plus the latest grad_norm. Stays single-line so
        ``nohup ... > log`` produces a readable progress trail.
        """
        eta_str = _fmt_hms(eta_s) if eta_s is not None else "--:--:--"
        bits = [
            f"step={self._step}/{self.total_steps}",
            f"elapsed={_fmt_hms(elapsed_s)}",
            f"eta={eta_str}",
            f"lr={self._lr:.2e}",
            f"img/s={self._throughput:.1f}",
            f"gn={self._grad_norm:.2f}",
        ]
        for k in ("total", "loss", "l1_rgb"):
            tr = self._losses.get(k)
            if tr is not None and tr.short.value is not None:
                bits.append(f"{k}={tr.short.value:.4f}")
                break
        for axis in self.task_names:
            psnr_tr = self._psnr.get(axis)
            lpips_tr = self._lpips.get(axis)
            psnr_v = psnr_tr.short.value if psnr_tr else None
            lpips_v = lpips_tr.short.value if lpips_tr else None
            if psnr_v is None and lpips_v is None:
                continue
            psnr_str = f"{psnr_v:.1f}dB" if psnr_v is not None else "—"
            lpips_str = f"{lpips_v:.3f}" if lpips_v is not None else "—"
            bits.append(f"{axis}={psnr_str}/{lpips_str}")
        print(" ".join(bits), flush=True)

    def note_preview(self, msg: str) -> None:
        self._last_preview = msg
        # Live picks up the new text on the next refresh; only headless
        # needs an explicit print.
        if self.headless:
            print(f"[preview] {msg}", flush=True)

    def render(self) -> Layout:
        # Refresh the wall-clock timer on every call. Live invokes this
        # at refresh_per_second Hz, so elapsed/ETA tick smoothly even
        # while we wait minutes between log_every ticks.
        elapsed_s = time.perf_counter() - self._t0
        eta_s = self._eta_seconds(elapsed_s, self._step)
        self._progress.update(
            self._task_id, completed=self._step,
            elapsed_str=_fmt_hms(elapsed_s),
            eta_str=_fmt_hms(eta_s) if eta_s is not None else "--:--:--",
        )
        layout = Layout()
        layout.split_column(
            Layout(Panel.fit(f"run: {self.run_name}", title="restora train"), size=3),
            Layout(self._progress, size=3),
            Layout(name="middle", size=22),
            Layout(Panel.fit(self._last_preview or "(no preview yet)", title="last preview"), size=3),
        )
        layout["middle"].split_row(
            self._losses_panel(), self._metrics_panel(), self._gpu_panel())
        return layout

    def _losses_panel(self) -> Panel:
        t = Table.grid(padding=(0, 1))
        t.add_column("loss")
        t.add_column("value", justify="right")
        t.add_column("trend", justify="right")
        for name, tr in self._losses.items():
            s = tr.short.value or 0.0
            l = tr.long.value or s
            arrow = "▼" if s < l else "▲"
            t.add_row(name, f"{s:.4f}", f"{arrow} {abs(s - l):.4f}")
        t.add_row("lr", f"{self._lr:.2e}", "")
        t.add_row("img/s", f"{self._throughput:.1f}", "")
        t.add_row("grad_norm", f"{self._grad_norm:.3f}", "")
        return Panel(t, title="losses (EMA)")

    def _metrics_panel(self) -> Panel:
        """Per-axis metrics: PSNR (higher better) + LPIPS (lower better).

        Trend arrows are direction-aware: PSNR ▲ = improving, LPIPS ▼ =
        improving. Both use long-EMA as the reference baseline so the
        arrow tracks "is short-EMA pulling away in the good direction?"
        """
        t = Table.grid(padding=(0, 1))
        t.add_column("task")
        t.add_column("PSNR", justify="right")
        t.add_column("Δ", justify="left")
        t.add_column("LPIPS", justify="right")
        t.add_column("Δ", justify="left")
        for name in self.task_names:
            psnr_tr = self._psnr.get(name)
            lpips_tr = self._lpips.get(name)
            psnr_s = psnr_tr.short.value if psnr_tr else None
            psnr_l = psnr_tr.long.value if psnr_tr else None
            lpips_s = lpips_tr.short.value if lpips_tr else None
            lpips_l = lpips_tr.long.value if lpips_tr else None
            if psnr_s is None:
                psnr_str, psnr_trend = "—", ""
            else:
                base = psnr_l if psnr_l is not None else psnr_s
                arrow = "▲" if psnr_s > base else "▼"
                psnr_str = f"{psnr_s:.1f}dB"
                psnr_trend = f"{arrow}{abs(psnr_s - base):.2f}"
            if lpips_s is None:
                lpips_str, lpips_trend = "—", ""
            else:
                base = lpips_l if lpips_l is not None else lpips_s
                arrow = "▼" if lpips_s < base else "▲"
                lpips_str = f"{lpips_s:.3f}"
                lpips_trend = f"{arrow}{abs(lpips_s - base):.3f}"
            t.add_row(name, psnr_str, psnr_trend, lpips_str, lpips_trend)
        return Panel(t, title="per-task metrics")

    def _gpu_panel(self) -> Panel:
        s = gpu_stats(0)
        if s is None:
            return Panel("gpu stats unavailable", title="gpu")
        t = Table.grid(padding=(0, 1))
        t.add_column()
        t.add_column(justify="right")
        t.add_row("name", s.name[:24])
        t.add_row("mem", f"{s.mem_used_gb:.1f}/{s.mem_total_gb:.1f} GB")
        t.add_row("util", f"{s.util_pct}%")
        t.add_row("temp", f"{s.temp_c}°C")
        t.add_row("pwr", f"{s.power_w:.0f}/{s.power_limit_w:.0f} W")
        return Panel(t, title="gpu")
