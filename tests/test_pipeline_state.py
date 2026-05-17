"""Tests for pipeline_state persistence + lookups."""
from pathlib import Path

import pytest

from restora_models.train.pipeline_state import PipelineState, STAGE_ORDER


def test_fresh_state_no_stages_complete(tmp_path):
    s = PipelineState(tmp_path)
    for stage in STAGE_ORDER:
        assert not s.is_complete(stage)


def test_mark_and_query(tmp_path):
    s = PipelineState(tmp_path)
    s.mark_complete("flow_distill", checkpoint=tmp_path / "flow.pt")
    assert s.is_complete("flow_distill")
    assert s.checkpoint_for("flow_distill") == tmp_path / "flow.pt"


def test_persistence_across_instances(tmp_path):
    s1 = PipelineState(tmp_path)
    s1.mark_complete("flow_distill", checkpoint=tmp_path / "flow.pt")
    s2 = PipelineState(tmp_path)
    assert s2.is_complete("flow_distill")


def test_next_pending(tmp_path):
    s = PipelineState(tmp_path)
    assert s.next_pending() == "flow_distill"
    s.mark_complete("flow_distill", checkpoint=tmp_path / "flow.pt")
    assert s.next_pending() == "backbone"


def test_reset_from(tmp_path):
    s = PipelineState(tmp_path)
    for st in ["flow_distill", "backbone", "refine"]:
        s.mark_complete(st, checkpoint=tmp_path / f"{st}.pt")
    s.reset_from("backbone")
    assert s.is_complete("flow_distill")
    assert not s.is_complete("backbone")
    assert not s.is_complete("refine")


def test_unknown_stage_raises(tmp_path):
    s = PipelineState(tmp_path)
    with pytest.raises(KeyError):
        s.is_complete("no_such_stage")
