"""Submission export format tests.

These tests intentionally avoid tempfile-based scratch directories.
On this Windows/sandbox setup, failed cleanup of system temp folders can leave
locked ``tmp*`` directories in the repository root.
"""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

from pcl_detection.data_pipeline import load_dataset_bundle
from pcl_detection.training_pipeline import _validate_submission_predictions


class ExportTests(unittest.TestCase):
    def test_existing_submission_files_match_spec_format(self) -> None:
        bundle = load_dataset_bundle(ROOT / "data")
        dev_lines = (ROOT / "dev.txt").read_text(encoding="utf-8").splitlines()
        test_lines = (ROOT / "test.txt").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(dev_lines), len(bundle.dev))
        self.assertEqual(len(test_lines), len(bundle.test))
        self.assertTrue(all(line in {"0", "1"} for line in dev_lines))
        self.assertTrue(all(line in {"0", "1"} for line in test_lines))

    def test_validate_submission_predictions_rejects_non_binary_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            _validate_submission_predictions([0, 2, 1])

    def test_validate_submission_predictions_rejects_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "one prediction per input example"):
            _validate_submission_predictions([0, 1], expected_count=3)


if __name__ == "__main__":
    unittest.main()
