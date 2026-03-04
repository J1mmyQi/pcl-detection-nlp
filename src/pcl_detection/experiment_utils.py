"""Helpers for comparing saved experiment runs."""

from __future__ import annotations

import json
from pathlib import Path


def collect_experiment_metrics(artifacts_root: str | Path = "artifacts") -> list[dict[str, object]]:
    root = Path(artifacts_root)
    rows: list[dict[str, object]] = []
    if not root.exists():
        return rows

    for candidate in sorted(root.iterdir()):
        metrics_path = candidate / "metrics.json"
        if not candidate.is_dir() or not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": candidate.name,
                "artifacts_dir": str(candidate),
                "precision": float(payload.get("precision", 0.0)),
                "recall": float(payload.get("recall", 0.0)),
                "f1": float(payload.get("f1", 0.0)),
                "tp": int(payload.get("tp", 0)),
                "fp": int(payload.get("fp", 0)),
                "fn": int(payload.get("fn", 0)),
                "tn": int(payload.get("tn", 0)),
            }
        )
    return rows


def rank_experiments(
    artifacts_root: str | Path = "artifacts",
    *,
    limit: int | None = None,
) -> list[dict[str, object]]:
    rows = collect_experiment_metrics(artifacts_root)
    rows.sort(key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])), reverse=True)
    if limit is not None:
        return rows[:limit]
    return rows


def write_experiment_summary(
    artifacts_root: str | Path = "artifacts",
    *,
    output_file: str | Path | None = None,
    limit: int | None = None,
) -> Path:
    rows = rank_experiments(artifacts_root, limit=limit)
    target = Path(output_file) if output_file is not None else Path(artifacts_root) / "experiment_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return target
