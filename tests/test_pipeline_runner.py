"""Tests for the pipeline runner (with fake stage runners)."""
from pathlib import Path

from restora_models.train.pipeline import run_pipeline, STAGE_RUNNERS
from restora_models.train.pipeline_state import PipelineState, STAGE_ORDER


def test_runner_executes_stages_in_order(tmp_path, monkeypatch):
    calls: list[str] = []
    def fake(name):
        def runner(out_dir, prev_checkpoint, config_path, **kw):
            calls.append(name)
            out_dir.mkdir(parents=True, exist_ok=True)
            ck = out_dir / "final.pt"
            ck.write_text("fake")
            return ck
        return runner
    monkeypatch.setattr("restora_models.train.pipeline.STAGE_RUNNERS",
                       {n: fake(n) for n in STAGE_ORDER})
    run_pipeline(run_root=tmp_path, config_path=None)
    assert calls == list(STAGE_ORDER)


def test_runner_skips_completed(tmp_path, monkeypatch):
    s = PipelineState(tmp_path)
    s.mark_complete("flow_distill", checkpoint=tmp_path / "flow.pt")
    s.mark_complete("backbone", checkpoint=tmp_path / "back.pt")
    calls: list[str] = []
    def fake(name):
        def runner(out_dir, prev_checkpoint, config_path, **kw):
            calls.append(name)
            out_dir.mkdir(parents=True, exist_ok=True)
            ck = out_dir / "final.pt"; ck.write_text("fake")
            return ck
        return runner
    monkeypatch.setattr("restora_models.train.pipeline.STAGE_RUNNERS",
                       {n: fake(n) for n in STAGE_ORDER})
    run_pipeline(run_root=tmp_path, config_path=None)
    assert "flow_distill" not in calls
    assert calls[0] == "refine"


def test_runner_extend_from_resets_subsequent(tmp_path, monkeypatch):
    s = PipelineState(tmp_path)
    for st in ["flow_distill", "backbone", "refine", "end_to_end"]:
        s.mark_complete(st, checkpoint=tmp_path / f"{st}.pt")
    calls: list[str] = []
    def fake(name):
        def runner(out_dir, prev_checkpoint, config_path, **kw):
            calls.append(name)
            out_dir.mkdir(parents=True, exist_ok=True)
            ck = out_dir / "final.pt"; ck.write_text("fake")
            return ck
        return runner
    monkeypatch.setattr("restora_models.train.pipeline.STAGE_RUNNERS",
                       {n: fake(n) for n in STAGE_ORDER})
    run_pipeline(run_root=tmp_path, config_path=None, extend_from="backbone")
    assert "flow_distill" not in calls
    assert calls[0] == "backbone"
