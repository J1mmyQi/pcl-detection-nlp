"""Training, evaluation, and submission export helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .config import ExperimentConfig, load_config
from .models import RobertaClassifierScaffold, TfidfSvmClassifier
from .data_pipeline import DatasetBundle, PCLRecord, load_dataset_bundle

SUBMISSION_LABELS = {0, 1}


@dataclass(slots=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> BinaryMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    tp = fp = fn = tn = 0
    for expected, predicted in zip(y_true, y_pred, strict=True):
        if expected == 1 and predicted == 1:
            tp += 1
        elif expected == 0 and predicted == 1:
            fp += 1
        elif expected == 1 and predicted == 0:
            fn += 1
        elif expected == 0 and predicted == 0:
            tn += 1
        else:
            raise ValueError(f"Unsupported binary labels: expected={expected}, predicted={predicted}")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return BinaryMetrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def collect_error_examples(records: list[PCLRecord], predictions: list[int], limit: int = 25) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for record, predicted in zip(records, predictions, strict=True):
        if record.binary_label == predicted:
            continue
        examples.append(
            {
                "record_id": record.record_id,
                "keyword": record.keyword,
                "gold": record.binary_label,
                "pred": predicted,
                "text": record.text,
            }
        )
        if len(examples) >= limit:
            break
    return examples


def write_prediction_file(predictions: list[int], output_file: str | Path) -> Path:
    _validate_submission_predictions(predictions)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(prediction)) for prediction in predictions) + "\n", encoding="utf-8")
    return path


def train_from_config(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    bundle = load_dataset_bundle(config.data_dir)
    model = _build_model(config)

    train_texts = [record.text for record in bundle.train]
    train_labels = [int(record.binary_label) for record in bundle.train]
    dev_texts = [record.text for record in bundle.dev]
    dev_labels = [int(record.binary_label) for record in bundle.dev]

    model.fit(train_texts, train_labels)
    predictions = model.predict(dev_texts)

    metrics = compute_binary_metrics(dev_labels, predictions)
    errors = collect_error_examples(bundle.dev, predictions)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.artifacts_dir)
    (config.artifacts_dir / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    (config.artifacts_dir / "dev_error_examples.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    return {
        "model_name": config.model_name,
        "artifacts_dir": str(config.artifacts_dir),
        "metrics": metrics.to_dict(),
        "error_count": len(errors),
    }


def export_split_predictions(
    config_path: str | Path,
    model_dir: str | Path,
    split: str,
    output_file: str | Path,
) -> Path:
    config = load_config(config_path)
    bundle = load_dataset_bundle(config.data_dir)
    model = _load_model(config.model_name, model_dir)
    records = bundle.get_split(split)
    predictions = model.predict([record.text for record in records])
    _validate_submission_predictions(predictions, expected_count=len(records))
    return write_prediction_file(predictions, output_file)


def export_predictions_from_trained_config(
    config_path: str | Path,
    split: str,
    output_file: str | Path,
) -> Path:
    config = load_config(config_path)
    return export_split_predictions(
        config_path=config_path,
        model_dir=config.artifacts_dir,
        split=split,
        output_file=output_file,
    )


def _validate_submission_predictions(predictions: list[int], expected_count: int | None = None) -> None:
    if expected_count is not None and len(predictions) != expected_count:
        raise ValueError(
            "Submission files must contain exactly one prediction per input example: "
            f"expected {expected_count}, got {len(predictions)}."
        )

    for index, prediction in enumerate(predictions, start=1):
        label = int(prediction)
        if label not in SUBMISSION_LABELS:
            raise ValueError(
                "Submission files must contain exactly one binary prediction (0 or 1) per line: "
                f"line {index} has {prediction!r}."
            )


def _build_model(config: ExperimentConfig):
    if config.model_name == "tfidf_svm":
        raw_char_min = config.tfidf.get("char_ngram_min")
        raw_char_max = config.tfidf.get("char_ngram_max")
        char_ngram_range = None
        if raw_char_min is not None and raw_char_max is not None:
            char_ngram_range = (int(raw_char_min), int(raw_char_max))

        return TfidfSvmClassifier(
            ngram_range=(int(config.tfidf.get("ngram_min", 1)), int(config.tfidf.get("ngram_max", 2))),
            min_df=int(config.tfidf.get("min_df", 2)),
            max_features=config.tfidf.get("max_features"),
            sublinear_tf=bool(config.tfidf.get("sublinear_tf", False)),
            stop_words=config.tfidf.get("stop_words"),
            char_ngram_range=char_ngram_range,
            char_max_features=config.tfidf.get("char_max_features", config.tfidf.get("max_features")),
            c=float(config.svm.get("c", 1.0)),
            class_weight=config.svm.get("class_weight"),
        )
    if config.model_name == "roberta":
        return RobertaClassifierScaffold(
            pretrained_model_name=str(config.roberta.get("pretrained_model_name", "roberta-base")),
            max_length=int(config.roberta.get("max_length", 256)),
            learning_rate=float(config.roberta.get("learning_rate", 2e-5)),
            batch_size=int(config.roberta.get("batch_size", 16)),
            epochs=int(config.roberta.get("epochs", 3)),
        )
    raise ValueError(f"Unsupported model_name: {config.model_name}")


def _load_model(model_name: str, model_dir: str | Path):
    if model_name == "tfidf_svm":
        return TfidfSvmClassifier.load(model_dir)
    if model_name == "roberta":
        return RobertaClassifierScaffold.load(model_dir)
    raise ValueError(f"Unsupported model_name: {model_name}")
