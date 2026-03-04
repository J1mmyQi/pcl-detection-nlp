"""Evaluation helper tests."""

import unittest

from pcl_detection.training_pipeline import compute_binary_metrics


class MetricsTests(unittest.TestCase):
    def test_compute_binary_metrics(self) -> None:
        metrics = compute_binary_metrics(
            y_true=[1, 0, 1, 0],
            y_pred=[1, 1, 0, 0],
        )
        self.assertEqual(metrics.tp, 1)
        self.assertEqual(metrics.fp, 1)
        self.assertEqual(metrics.fn, 1)
        self.assertEqual(metrics.tn, 1)
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1, 0.5)


if __name__ == "__main__":
    unittest.main()
