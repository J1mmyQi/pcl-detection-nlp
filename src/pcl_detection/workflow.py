"""Compatibility layer exposing the old workflow API.

Data loading and EDA code now lives in ``pcl_detection.data_pipeline``.
Part 3 code now lives in ``pcl_detection.models``.
Training and export code now lives in ``pcl_detection.training_pipeline``.
"""

from __future__ import annotations

from .data_pipeline import DEFAULT_DATA_DIR
from .data_pipeline import DatasetBundle, PCLRecord
from .data_pipeline import NEGATIVE_LABELS, POSITIVE_LABELS
from .data_pipeline import build_eda_report, load_dataset_bundle, run_stats, to_binary_label
from .training_pipeline import BinaryMetrics
from .training_pipeline import _validate_submission_predictions
from .training_pipeline import collect_error_examples, compute_binary_metrics
from .training_pipeline import export_split_predictions, train_from_config, write_prediction_file

__all__ = [
    "BinaryMetrics",
    "DEFAULT_DATA_DIR",
    "DatasetBundle",
    "NEGATIVE_LABELS",
    "PCLRecord",
    "POSITIVE_LABELS",
    "_validate_submission_predictions",
    "build_eda_report",
    "collect_error_examples",
    "compute_binary_metrics",
    "export_split_predictions",
    "load_dataset_bundle",
    "run_stats",
    "to_binary_label",
    "train_from_config",
    "write_prediction_file",
]
