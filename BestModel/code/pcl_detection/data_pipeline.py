"""Data loading, preprocessing, and EDA helpers."""

from __future__ import annotations

import ast
from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path

from .config import load_config

DEFAULT_DATA_DIR = Path("data")
RAW_TSV_NAME = "dontpatronizeme_pcl.tsv"
TRAIN_LABELS_NAME = "train_semeval_parids-labels.csv"
DEV_LABELS_NAME = "dev_semeval_parids-labels.csv"
TEST_TSV_NAME = "task4_test.tsv"

NEGATIVE_LABELS = {0, 1}
POSITIVE_LABELS = {2, 3, 4}


@dataclass(slots=True)
class PCLRecord:
    record_id: str
    article_id: str
    keyword: str
    country_code: str
    text: str
    original_label: int | None = None
    binary_label: int | None = None
    category_labels: tuple[int, ...] | None = None
    split: str = "unspecified"


@dataclass(slots=True)
class DatasetBundle:
    train: list[PCLRecord]
    dev: list[PCLRecord]
    test: list[PCLRecord]

    def get_split(self, split: str) -> list[PCLRecord]:
        if split == "train":
            return self.train
        if split == "dev":
            return self.dev
        if split == "test":
            return self.test
        raise ValueError(f"Unsupported split: {split}")


def load_dataset_bundle(data_dir: str | Path = DEFAULT_DATA_DIR) -> DatasetBundle:
    root = Path(data_dir)
    labeled_records = _load_labeled_records(root / RAW_TSV_NAME)
    train_category_map = _load_category_labels(root / TRAIN_LABELS_NAME)
    dev_category_map = _load_category_labels(root / DEV_LABELS_NAME)

    train_records = [
        _copy_record(labeled_records[record_id], split="train", category_labels=category_labels)
        for record_id, category_labels in train_category_map.items()
    ]
    dev_records = [
        _copy_record(labeled_records[record_id], split="dev", category_labels=category_labels)
        for record_id, category_labels in dev_category_map.items()
    ]
    test_records = _load_test_records(root / TEST_TSV_NAME)
    return DatasetBundle(train=train_records, dev=dev_records, test=test_records)


def to_binary_label(original_label: int) -> int:
    if original_label in NEGATIVE_LABELS:
        return 0
    if original_label in POSITIVE_LABELS:
        return 1
    raise ValueError(f"Unexpected original label: {original_label}")


def build_eda_report(bundle: DatasetBundle) -> dict[str, object]:
    return {
        "train": _split_profile(bundle.train),
        "dev": _split_profile(bundle.dev),
        "test": _split_profile(bundle.test),
        "top_train_keywords": _top_keywords(bundle.train),
        "top_positive_keywords": _top_keywords([record for record in bundle.train if record.binary_label == 1]),
    }


def run_stats(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    bundle = load_dataset_bundle(config.data_dir)
    report = build_eda_report(bundle)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (config.artifacts_dir / "eda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_labeled_records(path: Path) -> dict[str, PCLRecord]:
    records: dict[str, PCLRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line or line.startswith("-") or line.startswith("The Don"):
                continue
            parts = line.split("\t")
            if len(parts) != 6 or not parts[0].isdigit():
                continue
            record_id, article_id, keyword, country_code, text, label = parts
            original_label = int(label)
            records[record_id] = PCLRecord(
                record_id=record_id,
                article_id=article_id,
                keyword=keyword,
                country_code=country_code,
                text=text.strip(),
                original_label=original_label,
                binary_label=to_binary_label(original_label),
                split="full_labeled",
            )
    return records


def _load_test_records(path: Path) -> list[PCLRecord]:
    records: list[PCLRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                continue
            record_id, article_id, keyword, country_code, text = parts
            records.append(
                PCLRecord(
                    record_id=record_id,
                    article_id=article_id,
                    keyword=keyword,
                    country_code=country_code,
                    text=text.strip(),
                    split="test",
                )
            )
    return records


def _load_category_labels(path: Path) -> dict[str, tuple[int, ...]]:
    category_map: dict[str, tuple[int, ...]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_labels = ast.literal_eval(row["label"])
            category_map[str(row["par_id"])] = tuple(int(value) for value in raw_labels)
    return category_map


def _copy_record(record: PCLRecord, *, split: str, category_labels: tuple[int, ...] | None) -> PCLRecord:
    return PCLRecord(
        record_id=record.record_id,
        article_id=record.article_id,
        keyword=record.keyword,
        country_code=record.country_code,
        text=record.text,
        original_label=record.original_label,
        binary_label=record.binary_label,
        category_labels=category_labels,
        split=split,
    )


def _split_profile(records: list[PCLRecord]) -> dict[str, object]:
    token_lengths = [len(record.text.split()) for record in records]
    binary_counter = Counter(record.binary_label for record in records if record.binary_label is not None)
    return {
        "size": len(records),
        "binary_label_distribution": dict(binary_counter),
        "avg_tokens": round(sum(token_lengths) / len(token_lengths), 2) if token_lengths else 0.0,
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
    }


def _top_keywords(records: list[PCLRecord], top_k: int = 10) -> list[tuple[str, int]]:
    return Counter(record.keyword for record in records).most_common(top_k)
