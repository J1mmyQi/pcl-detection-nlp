"""Dataset loading smoke tests."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

from pcl_detection.data_pipeline import load_dataset_bundle, to_binary_label


class DataLoadingTests(unittest.TestCase):
    def test_binary_label_conversion(self) -> None:
        self.assertEqual(to_binary_label(0), 0)
        self.assertEqual(to_binary_label(1), 0)
        self.assertEqual(to_binary_label(2), 1)
        self.assertEqual(to_binary_label(4), 1)

    def test_load_dataset_bundle(self) -> None:
        bundle = load_dataset_bundle(Path("data"))
        self.assertGreater(len(bundle.train), 0)
        self.assertGreater(len(bundle.dev), 0)
        self.assertGreater(len(bundle.test), 0)
        self.assertTrue(all(record.binary_label in {0, 1} for record in bundle.train))
        self.assertTrue(all(record.binary_label in {0, 1} for record in bundle.dev))
        self.assertTrue(all(record.binary_label is None for record in bundle.test))


if __name__ == "__main__":
    unittest.main()
