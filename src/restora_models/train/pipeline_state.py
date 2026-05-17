"""Persistent state file for the multi-stage training orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

STAGE_ORDER = (
    "flow_distill",
    "backbone",
    "refine",
    "end_to_end",
    "distill_small",
    "distill_medium",
    "distill_nano",
)


class PipelineState:
    FILE_NAME = "pipeline_state.json"

    def __init__(self, run_root: Path | str):
        self.root = Path(run_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILE_NAME
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            data.setdefault("stages", {})
            return data
        return {"stages": {}, "version": 1}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def is_complete(self, stage: str) -> bool:
        if stage not in STAGE_ORDER:
            raise KeyError(f"unknown stage {stage!r}; must be in {STAGE_ORDER}")
        return bool(self._data["stages"].get(stage, {}).get("complete", False))

    def checkpoint_for(self, stage: str) -> Path | None:
        entry = self._data["stages"].get(stage, {})
        ckpt = entry.get("checkpoint")
        return Path(ckpt) if ckpt else None

    def mark_complete(self, stage: str, *, checkpoint: Path) -> None:
        if stage not in STAGE_ORDER:
            raise KeyError(f"unknown stage {stage!r}")
        self._data["stages"][stage] = {"complete": True, "checkpoint": str(checkpoint)}
        self._save()

    def next_pending(self) -> str | None:
        for stage in STAGE_ORDER:
            if not self.is_complete(stage):
                return stage
        return None

    def reset_from(self, stage: str) -> None:
        if stage not in STAGE_ORDER:
            raise KeyError(f"unknown stage {stage!r}")
        idx = STAGE_ORDER.index(stage)
        for st in STAGE_ORDER[idx:]:
            self._data["stages"][st] = {"complete": False, "checkpoint": None}
        self._save()
