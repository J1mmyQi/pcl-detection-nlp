"""Config loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    data_dir: Path
    artifacts_dir: Path
    model_name: str
    random_state: int
    tfidf: dict[str, Any]
    svm: dict[str, Any]
    roberta: dict[str, Any]


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        data_dir=Path(payload["data_dir"]),
        artifacts_dir=Path(payload["artifacts_dir"]),
        model_name=payload["model_name"],
        random_state=int(payload.get("random_state", 42)),
        tfidf=dict(payload.get("tfidf", {})),
        svm=dict(payload.get("svm", {})),
        roberta=dict(payload.get("roberta", {})),
    )
