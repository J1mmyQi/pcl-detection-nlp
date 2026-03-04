from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

from pcl_detection.experiment_utils import rank_experiments


class ExperimentSummaryTests(unittest.TestCase):
    def test_rank_experiments_returns_sorted_rows(self) -> None:
        rows = rank_experiments(ROOT / "artifacts", limit=5)
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("experiment", rows[0])
        self.assertIn("f1", rows[0])
        for earlier, later in zip(rows, rows[1:], strict=False):
            self.assertGreaterEqual(float(earlier["f1"]), float(later["f1"]))


if __name__ == "__main__":
    unittest.main()
