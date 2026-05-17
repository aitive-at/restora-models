"""Factory to build a VideoWindowDataset from a list of source specs."""
from __future__ import annotations

from typing import Any, Callable, Sequence

from restora_models.data.compound_wrapper import CompoundDegradationWrapper
from restora_models.data.reds import REDSDataset
from restora_models.data.video_window import VideoSubDataset, VideoWindowDataset
from restora_models.data.vimeo_septuplet import VimeoSeptupletDataset

_BUILDERS: dict[str, Callable[[dict], VideoSubDataset]] = {
    "reds": lambda kw: REDSDataset(**{k: v for k, v in kw.items() if k not in ("type", "weight")}),
    "vimeo_septuplet": lambda kw: VimeoSeptupletDataset(**{k: v for k, v in kw.items() if k not in ("type", "weight")}),
}


def build_video_window_dataset(sources: Sequence[dict]) -> VideoWindowDataset:
    """Build the composite dataset from a list of source dicts.

    Each entry has a `type` key matching a registered builder, plus
    arbitrary kwargs forwarded to the sub-dataset constructor, plus an
    optional `weight` (default 1.0) used by sample_random().
    """
    subs: list[VideoSubDataset] = []
    weights: list[float] = []
    for s in sources:
        kind = s.get("type")
        if kind not in _BUILDERS:
            raise KeyError(
                f"unknown video source type {kind!r}; have {sorted(_BUILDERS)}"
            )
        subs.append(_BUILDERS[kind](s))
        weights.append(float(s.get("weight", 1.0)))
    return VideoWindowDataset(sub_datasets=subs, weights=weights)


def build_compound_video_dataset(
    sources: Sequence[dict],
    *,
    data_cfg: Any,
    seed: int = 0,
) -> CompoundDegradationWrapper:
    """Build the composite dataset *and* wrap it with the compound degradation pipeline.

    Use this from the trainer so all CPU degradation work runs inside
    DataLoader workers. ``data_cfg`` is the same ``cfg.data`` object the
    trainer already has (DataConfig); we only read film-overlay /
    film-color-cast / gate-weave / mpeg knobs off it.
    """
    inner = build_video_window_dataset(sources)
    return CompoundDegradationWrapper(inner, data_cfg=data_cfg, seed=seed)
