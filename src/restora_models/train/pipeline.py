"""End-to-end multi-stage training orchestrator.

Each stage is a callable that returns the path to its final checkpoint.
The runner loads PipelineState from run_root, executes pending stages
in order, and persists state after each completes.

Missing stages (flow_distill until Phase 17, distill_* until Phase 14)
emit a clear message and skip; the pipeline marks them complete with a
None checkpoint so the rest can proceed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from restora_models.train.pipeline_state import PipelineState, STAGE_ORDER


StageRunner = Callable[..., Optional[Path]]


def _stage_train(out_dir: Path, prev_checkpoint: Path | None,
                 config_path: Path | None, **kw) -> Path:
    """Stages 1-3 all call into the trainer with different freeze/warm-start args.

    Future enhancement: pass `warm_start` / `freeze` / `lr_scale` through to
    run_train_stage once it supports them (Phase 14 / Phase 18 polish).
    """
    from restora_models.train.trainer import run_train_stage
    return run_train_stage(
        out_dir=out_dir,
        config_path=config_path,
        flow_estimator_ckpt=kw.get("flow_estimator_ckpt"),
        warm_start=prev_checkpoint,
        freeze=kw.get("freeze", ()),
        lr_scale=kw.get("lr_scale", 1.0),
    )


def _stage_flow_distill(out_dir: Path, prev_checkpoint, config_path, **kw) -> Path | None:
    try:
        from restora_models.train.flow_distill import run_flow_distill
    except ImportError:
        print(f"[pipeline] flow_distill: module not implemented (Phase 17 pending); skipping")
        return None
    return run_flow_distill(out_dir=out_dir, config_path=config_path)


def _stage_backbone(out_dir, prev_checkpoint, config_path, **kw) -> Path:
    return _stage_train(out_dir, prev_checkpoint, config_path,
                        freeze=("refine",), flow_estimator_ckpt=prev_checkpoint)


def _stage_refine(out_dir, prev_checkpoint, config_path, **kw) -> Path:
    return _stage_train(out_dir, prev_checkpoint, config_path,
                        warm_start=prev_checkpoint, freeze=("flow_estimator", "backbone"))


def _stage_end_to_end(out_dir, prev_checkpoint, config_path, **kw) -> Path:
    return _stage_train(out_dir, prev_checkpoint, config_path,
                        warm_start=prev_checkpoint, freeze=("flow_estimator",),
                        lr_scale=0.1)


def _make_distill_runner(size: str):
    def runner(out_dir: Path, prev_checkpoint: Path | None,
               config_path: Path | None, **kw) -> Path | None:
        try:
            from restora_models.train.distill import run_distill
        except ImportError:
            print(f"[pipeline] distill_{size}: module not implemented (Phase 14 pending); skipping")
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        return run_distill(
            teacher=prev_checkpoint, output=out_dir / "final.pt",
            data=None, student_preset=size,
        )
    return runner


STAGE_RUNNERS: dict[str, StageRunner] = {
    "flow_distill":   _stage_flow_distill,
    "backbone":       _stage_backbone,
    "refine":         _stage_refine,
    "end_to_end":     _stage_end_to_end,
    "distill_small":  _make_distill_runner("small"),
    "distill_medium": _make_distill_runner("medium"),
    "distill_nano":   _make_distill_runner("nano"),
}


def run_pipeline(
    *,
    run_root: Path,
    config_path: Path | None,
    extend_from: str | None = None,
) -> None:
    """Run all pending stages in STAGE_ORDER.

    Args:
        run_root: Root for pipeline state + per-stage outputs.
        config_path: Training config YAML (forwarded to all stages).
        extend_from: If set, mark this stage and everything after as pending
            and rerun. Used when new training data is added.
    """
    state = PipelineState(run_root)
    if extend_from is not None:
        state.reset_from(extend_from)

    for stage in STAGE_ORDER:
        if state.is_complete(stage):
            print(f"[pipeline] {stage}: already complete; skipping")
            continue
        runner = STAGE_RUNNERS[stage]
        # Find the most recent completed predecessor's checkpoint
        prev_ckpt = None
        for prev in reversed(STAGE_ORDER[: STAGE_ORDER.index(stage)]):
            if state.is_complete(prev):
                prev_ckpt = state.checkpoint_for(prev)
                break
        out_dir = Path(run_root) / stage
        print(f"[pipeline] === {stage} ===  (prev ckpt: {prev_ckpt})")
        ckpt = runner(out_dir=out_dir, prev_checkpoint=prev_ckpt, config_path=config_path)
        # Even if runner returned None (not-implemented skip), mark complete
        # so the next stage can proceed; but record a None checkpoint so
        # downstream stages can detect there's nothing to warm-start from.
        state.mark_complete(stage, checkpoint=ckpt if ckpt is not None else (out_dir / "SKIPPED"))
