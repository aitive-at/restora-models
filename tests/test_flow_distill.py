"""Tests for the static-unroll RAFT student in models/flow_distill.py."""
import torch

from restora_models.models.flow_distill import FlowDistill


def test_flow_distill_output_shape():
    m = FlowDistill(iters=4).eval()
    pair = torch.randn(2, 2, 3, 128, 128)
    flow = m(pair)
    assert flow.shape == (2, 2, 128, 128), f"got {tuple(flow.shape)}"


def test_flow_distill_no_python_loop_in_graph():
    """The forward must not contain Python-level loops in the traced graph."""
    m = FlowDistill(iters=4).eval()
    pair = torch.randn(1, 2, 3, 64, 64)
    traced = torch.jit.trace(m, pair)
    assert "prim::Loop" not in str(traced.graph)


def test_flow_distill_param_budget():
    m = FlowDistill(iters=4)
    n = sum(p.numel() for p in m.parameters())
    assert 2_000_000 < n < 8_000_000, f"unexpected param count: {n}"
