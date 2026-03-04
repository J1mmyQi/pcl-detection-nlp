"""Helpers for running the project interactively from a notebook."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import unittest

from .experiment_utils import rank_experiments
from .training_pipeline import export_predictions_from_trained_config


@dataclass(slots=True)
class NotebookContext:
    root: Path
    best_config_path: Path


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("Could not find the repository root.")


def build_notebook_context(start: Path) -> NotebookContext:
    root = find_repo_root(start)
    return NotebookContext(
        root=root,
        best_config_path=root / "configs" / "best_artifact_ensemble.json",
    )


def run_unittest_targets(*targets: str) -> unittest.result.TestResult:
    loader = unittest.defaultTestLoader
    if targets:
        suite = unittest.TestSuite(loader.loadTestsFromName(target) for target in targets)
    else:
        suite = loader.discover("tests")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise AssertionError("One or more tests failed.")
    return result


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_training_artifacts(artifacts_dir: str | Path) -> dict[str, object]:
    root = Path(artifacts_dir)
    return {
        "metrics": load_json(root / "metrics.json"),
        "errors": load_json(root / "dev_error_examples.json"),
    }


def export_default_submission_files(config_path: str | Path, repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root)
    dev_path = export_predictions_from_trained_config(config_path, split="dev", output_file=root / "dev.txt")
    test_path = export_predictions_from_trained_config(config_path, split="test", output_file=root / "test.txt")

    dev_lines = dev_path.read_text(encoding="utf-8").splitlines()
    test_lines = test_path.read_text(encoding="utf-8").splitlines()
    return {
        "dev_path": str(dev_path),
        "test_path": str(test_path),
        "dev_line_count": len(dev_lines),
        "test_line_count": len(test_lines),
        "dev_preview": dev_lines[:5],
        "test_preview": test_lines[:5],
    }


def summarize_experiments(repo_root: str | Path, limit: int = 10) -> list[dict[str, object]]:
    root = Path(repo_root)
    return rank_experiments(root / "artifacts", limit=limit)
