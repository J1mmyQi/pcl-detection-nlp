"""Training, evaluation, and submission export helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, load_config
from .models import (
    ArtifactEnsembleClassifier,
    RobertaClassifierScaffold,
    RobertaProbeClassifier,
    TfidfSvmClassifier,
    WeightedEnsembleClassifier,
)
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

    train_texts = [record.text for record in bundle.train]
    train_labels = [int(record.binary_label) for record in bundle.train]
    dev_texts = [record.text for record in bundle.dev]
    dev_labels = [int(record.binary_label) for record in bundle.dev]

    threshold_summary: dict[str, Any] | None = None
    threshold_options = _get_threshold_tuning_options(config)
    if bool(threshold_options.get("enabled", False)):
        tuned_threshold, threshold_summary = _select_threshold(
            config,
            train_texts,
            train_labels,
            validation_fraction=float(threshold_options["validation_fraction"]),
        )
        _apply_threshold(config, tuned_threshold)

    model = _build_model(config)
    model.fit(train_texts, train_labels)
    predictions = model.predict(dev_texts)

    metrics = compute_binary_metrics(dev_labels, predictions)
    errors = collect_error_examples(bundle.dev, predictions)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.artifacts_dir)
    (config.artifacts_dir / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    (config.artifacts_dir / "dev_error_examples.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    if threshold_summary is not None:
        (config.artifacts_dir / "threshold_tuning.json").write_text(
            json.dumps(threshold_summary, indent=2),
            encoding="utf-8",
        )

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
            classifier_type=str(config.svm.get("classifier_type", "linear_svc")),
            threshold=float(config.svm.get("threshold", 0.0)),
        )
    if config.model_name == "roberta":
        return RobertaClassifierScaffold(
            pretrained_model_name=str(config.roberta.get("pretrained_model_name", "roberta-base")),
            max_length=int(config.roberta.get("max_length", 256)),
            learning_rate=float(config.roberta.get("learning_rate", 2e-5)),
            batch_size=int(config.roberta.get("batch_size", 16)),
            epochs=int(config.roberta.get("epochs", 3)),
            threshold=float(config.roberta.get("threshold", 0.0)),
            allow_download=bool(config.roberta.get("allow_download", False)),
            cache_dir=config.roberta.get("cache_dir"),
            huggingface_token=config.roberta.get("huggingface_token"),
            freeze_backbone=bool(config.roberta.get("freeze_backbone", False)),
            use_class_weights=bool(config.roberta.get("use_class_weights", True)),
            weight_decay=float(config.roberta.get("weight_decay", 0.01)),
            warmup_ratio=float(config.roberta.get("warmup_ratio", 0.1)),
        )
    if config.model_name == "roberta_probe":
        return RobertaProbeClassifier(
            pretrained_model_name=str(config.roberta.get("pretrained_model_name", "roberta-base")),
            max_length=int(config.roberta.get("max_length", 128)),
            batch_size=int(config.roberta.get("batch_size", 8)),
            threshold=float(config.roberta.get("threshold", 0.0)),
            allow_download=bool(config.roberta.get("allow_download", False)),
            cache_dir=config.roberta.get("cache_dir"),
            huggingface_token=config.roberta.get("huggingface_token"),
            c=float(config.roberta.get("probe_c", 1.0)),
            class_weight=config.roberta.get("class_weight", "balanced"),
        )
    if config.model_name == "artifact_ensemble":
        return ArtifactEnsembleClassifier(
            member_specs=list(config.ensemble.get("members", [])),
            weights=[float(value) for value in config.ensemble.get("weights", [])],
            threshold=float(config.ensemble.get("threshold", 0.0)),
            normalization=str(config.ensemble.get("normalization", "zscore")),
            calibrators=[
                {
                    "mean": float(item.get("mean", 0.0)),
                    "std": float(item.get("std", 1.0)),
                }
                for item in config.ensemble.get("calibrators", [])
            ],
        )
    if config.model_name == "tfidf_ensemble":
        member_specs = []
        raw_members = list(config.ensemble.get("members", []))
        for raw_member in raw_members:
            member_kind = str(raw_member.get("kind", "tfidf_svm"))
            tfidf = dict(raw_member.get("tfidf", {}))
            svm = dict(raw_member.get("svm", {}))
            raw_char_min = tfidf.get("char_ngram_min")
            raw_char_max = tfidf.get("char_ngram_max")
            char_ngram_range = None
            if raw_char_min is not None and raw_char_max is not None:
                char_ngram_range = (int(raw_char_min), int(raw_char_max))

            if member_kind == "tfidf_svm":
                member_specs.append(
                    {
                        "kind": "tfidf_svm",
                        "ngram_range": (int(tfidf.get("ngram_min", 1)), int(tfidf.get("ngram_max", 2))),
                        "min_df": int(tfidf.get("min_df", 2)),
                        "max_features": tfidf.get("max_features"),
                        "sublinear_tf": bool(tfidf.get("sublinear_tf", False)),
                        "stop_words": tfidf.get("stop_words"),
                        "char_ngram_range": char_ngram_range,
                        "char_max_features": tfidf.get("char_max_features", tfidf.get("max_features")),
                        "c": float(svm.get("c", 1.0)),
                        "class_weight": svm.get("class_weight"),
                        "classifier_type": str(svm.get("classifier_type", "linear_svc")),
                        "threshold": 0.0,
                    }
                )
                continue

            if member_kind == "latent_tfidf":
                latent = dict(raw_member.get("latent", {}))
                member_specs.append(
                    {
                        "kind": "latent_tfidf",
                        "ngram_range": (int(tfidf.get("ngram_min", 1)), int(tfidf.get("ngram_max", 2))),
                        "min_df": int(tfidf.get("min_df", 1)),
                        "max_features": tfidf.get("max_features"),
                        "sublinear_tf": bool(tfidf.get("sublinear_tf", True)),
                        "stop_words": tfidf.get("stop_words"),
                        "char_ngram_range": char_ngram_range,
                        "char_max_features": tfidf.get("char_max_features", tfidf.get("max_features")),
                        "n_components": int(latent.get("n_components", 150)),
                        "c": float(svm.get("c", 1.0)),
                        "class_weight": svm.get("class_weight"),
                        "threshold": float(latent.get("threshold", 0.5)),
                    }
                )
                continue

            raise ValueError(f"Unsupported ensemble member kind: {member_kind}")
        return WeightedEnsembleClassifier(
            member_specs=member_specs,
            weights=[float(weight) for weight in config.ensemble.get("weights", [])],
            threshold=float(config.ensemble.get("threshold", 0.0)),
        )
    raise ValueError(f"Unsupported model_name: {config.model_name}")


def _load_model(model_name: str, model_dir: str | Path):
    if model_name == "tfidf_svm":
        return TfidfSvmClassifier.load(model_dir)
    if model_name == "roberta":
        return RobertaClassifierScaffold.load(model_dir)
    if model_name == "roberta_probe":
        return RobertaProbeClassifier.load(model_dir)
    if model_name == "artifact_ensemble":
        return ArtifactEnsembleClassifier.load(model_dir)
    if model_name == "tfidf_ensemble":
        return WeightedEnsembleClassifier.load(model_dir)
    raise ValueError(f"Unsupported model_name: {model_name}")


def _select_threshold(
    config: ExperimentConfig,
    train_texts: list[str],
    train_labels: list[int],
    *,
    validation_fraction: float,
) -> tuple[float, dict[str, Any]]:
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        return 0.0, {"enabled": False, "reason": "scikit-learn is unavailable"}

    sub_train_texts, val_texts, sub_train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=validation_fraction,
        random_state=config.random_state,
        stratify=train_labels,
    )

    probe_config = ExperimentConfig(
        data_dir=config.data_dir,
        artifacts_dir=config.artifacts_dir,
        model_name=config.model_name,
        random_state=config.random_state,
        tfidf=dict(config.tfidf),
        svm=dict(config.svm),
        ensemble=dict(config.ensemble),
        roberta=dict(config.roberta),
    )
    _apply_threshold(probe_config, 0.0)

    model = _build_model(probe_config)
    model.fit(sub_train_texts, sub_train_labels)
    try:
        scores = model.decision_function(val_texts)
    except RuntimeError:
        return 0.0, {"enabled": False, "reason": "classifier does not expose decision scores"}

    candidate_thresholds = sorted({float(score) for score in scores})
    if not candidate_thresholds:
        return 0.0, {"enabled": False, "reason": "no candidate thresholds available"}

    best_threshold = 0.0
    best_metrics = compute_binary_metrics(val_labels, [1 if score >= 0.0 else 0 for score in scores])
    for threshold in candidate_thresholds:
        predictions = [1 if score >= threshold else 0 for score in scores]
        metrics = compute_binary_metrics(val_labels, predictions)
        if metrics.f1 > best_metrics.f1:
            best_threshold = threshold
            best_metrics = metrics

    summary = {
        "enabled": True,
        "validation_fraction": validation_fraction,
        "threshold": best_threshold,
        "validation_metrics": best_metrics.to_dict(),
    }
    return best_threshold, summary


def _get_threshold_tuning_options(config: ExperimentConfig) -> dict[str, Any]:
    if config.model_name in {"roberta", "roberta_probe"}:
        return {
            "enabled": bool(config.roberta.get("tune_threshold", False)),
            "validation_fraction": float(config.roberta.get("validation_fraction", 0.15)),
        }
    if config.model_name == "tfidf_ensemble":
        return {
            "enabled": bool(config.ensemble.get("tune_threshold", False)),
            "validation_fraction": float(config.ensemble.get("validation_fraction", 0.15)),
        }
    return {
        "enabled": bool(config.svm.get("tune_threshold", False)),
        "validation_fraction": float(config.svm.get("validation_fraction", 0.15)),
    }


def _apply_threshold(config: ExperimentConfig, threshold: float) -> None:
    if config.model_name in {"roberta", "roberta_probe"}:
        config.roberta["threshold"] = threshold
        return
    if config.model_name == "tfidf_ensemble":
        config.ensemble["threshold"] = threshold
        return
    config.svm["threshold"] = threshold
