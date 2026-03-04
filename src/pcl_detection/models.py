"""Part 3: Baseline model and proposed-approach model definitions."""

from __future__ import annotations

from collections import Counter
import json
import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class TextClassifier(Protocol):
    def fit(self, texts: list[str], labels: list[int]) -> None:
        """Train the classifier."""

    def predict(self, texts: list[str]) -> list[int]:
        """Predict binary labels."""

    def decision_function(self, texts: list[str]) -> list[float]:
        """Return raw decision scores when available."""

    def save(self, output_dir: str | Path) -> Path:
        """Persist model artifacts."""

    @classmethod
    def load(cls, model_dir: str | Path) -> "TextClassifier":
        """Reload model artifacts."""


@dataclass(slots=True)
class TfidfSvmClassifier:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_features: int | None = 30000
    sublinear_tf: bool = False
    stop_words: str | None = None
    char_ngram_range: tuple[int, int] | None = None
    char_max_features: int | None = 30000
    c: float = 1.0
    class_weight: str | None = "balanced"
    classifier_type: str = "linear_svc"
    threshold: float = 0.0
    backend: str | None = field(init=False, default=None)
    pipeline: object | None = field(init=False, default=None, repr=False)
    token_scores: dict[str, float] | None = field(init=False, default=None, repr=False)
    bias: float = field(init=False, default=0.0, repr=False)

    def fit(self, texts: list[str], labels: list[int]) -> None:
        try:
            from sklearn.pipeline import FeatureUnion
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.svm import LinearSVC
        except ImportError:
            self._fit_fallback(texts, labels)
            return

        vectorizers: list[tuple[str, object]] = [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_features=self.max_features,
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words=self.stop_words,
                    sublinear_tf=self.sublinear_tf,
                ),
            )
        ]
        if self.char_ngram_range is not None:
            vectorizers.append(
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=self.char_ngram_range,
                        min_df=self.min_df,
                        max_features=self.char_max_features,
                        lowercase=True,
                        strip_accents="unicode",
                        sublinear_tf=self.sublinear_tf,
                    ),
                )
            )

        if self.classifier_type == "linear_svc":
            classifier = LinearSVC(
                C=self.c,
                class_weight=self.class_weight,
                random_state=42,
            )
        elif self.classifier_type == "logreg":
            classifier = LogisticRegression(
                C=self.c,
                class_weight=self.class_weight,
                random_state=42,
                max_iter=2000,
                solver="liblinear",
            )
        else:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")

        self.pipeline = Pipeline(
            steps=[
                (
                    "features",
                    FeatureUnion(transformer_list=vectorizers),
                ),
                ("classifier", classifier),
            ]
        )
        self.pipeline.fit(texts, labels)
        self.backend = "sklearn"

    def predict(self, texts: list[str]) -> list[int]:
        if self.backend == "sklearn" and self.pipeline is not None:
            if self.threshold != 0.0:
                return [1 if score >= self.threshold else 0 for score in self.decision_function(texts)]
            return [int(value) for value in self.pipeline.predict(texts).tolist()]
        if self.backend == "fallback" and self.token_scores is not None:
            return [self._predict_fallback(text) for text in texts]
        if self.pipeline is None and self.token_scores is None:
            raise RuntimeError("The model has not been trained or loaded yet.")
        raise RuntimeError("Model state is inconsistent.")

    def decision_function(self, texts: list[str]) -> list[float]:
        if self.backend == "sklearn" and self.pipeline is not None:
            if hasattr(self.pipeline, "decision_function"):
                scores = self.pipeline.decision_function(texts)
                if hasattr(scores, "tolist"):
                    return [float(value) for value in scores.tolist()]
                return [float(value) for value in scores]
            raise RuntimeError("The configured classifier does not expose decision scores.")
        if self.backend == "fallback":
            raise RuntimeError("Fallback model does not expose decision scores.")
        raise RuntimeError("The model has not been trained or loaded yet.")

    def save(self, output_dir: str | Path) -> Path:
        if self.pipeline is None and self.token_scores is None:
            raise RuntimeError("Cannot save an untrained model.")
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / "model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(self, handle)
        return model_path

    @classmethod
    def load(cls, model_dir: str | Path) -> "TfidfSvmClassifier":
        model_path = Path(model_dir) / "model.pkl"
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected model type in {model_path}")
        return model

    def _fit_fallback(self, texts: list[str], labels: list[int]) -> None:
        doc_freq = Counter()
        positive_counts = Counter()
        negative_counts = Counter()
        positive_docs = 0
        negative_docs = 0

        for text, label in zip(texts, labels, strict=True):
            tokens = self._tokenize(text)
            for token in set(tokens):
                doc_freq[token] += 1
            if label == 1:
                positive_docs += 1
                positive_counts.update(tokens)
            else:
                negative_docs += 1
                negative_counts.update(tokens)

        candidates = [token for token, count in doc_freq.items() if count >= self.min_df]
        vocab_size = max(len(candidates), 1)
        positive_total = sum(positive_counts[token] for token in candidates)
        negative_total = sum(negative_counts[token] for token in candidates)

        scores: dict[str, float] = {}
        for token in candidates:
            positive_prob = (positive_counts[token] + 1.0) / (positive_total + vocab_size)
            negative_prob = (negative_counts[token] + 1.0) / (negative_total + vocab_size)
            scores[token] = math.log(positive_prob) - math.log(negative_prob)

        if self.max_features is not None and len(scores) > self.max_features:
            ranked = sorted(scores.items(), key=lambda item: abs(item[1]), reverse=True)
            scores = dict(ranked[: self.max_features])

        self.token_scores = scores
        self.bias = math.log((positive_docs + 1.0) / (negative_docs + 1.0))
        self.backend = "fallback"

    def _predict_fallback(self, text: str) -> int:
        assert self.token_scores is not None
        score = self.bias
        for token in self._tokenize(text):
            score += self.token_scores.get(token, 0.0)
        return 1 if score >= 0.0 else 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in text.lower().split() if token]


@dataclass(slots=True)
class LatentTfidfClassifier:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_features: int | None = 25000
    sublinear_tf: bool = True
    stop_words: str | None = None
    char_ngram_range: tuple[int, int] | None = (3, 5)
    char_max_features: int | None = 40000
    n_components: int = 150
    c: float = 1.0
    class_weight: str | dict[int, float] | None = "balanced"
    threshold: float = 0.5
    pipeline: object | None = field(init=False, default=None, repr=False)

    def fit(self, texts: list[str], labels: list[int]) -> None:
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import FeatureUnion, Pipeline
            from sklearn.preprocessing import Normalizer
        except ImportError as exc:
            raise RuntimeError("LatentTfidfClassifier requires scikit-learn.") from exc

        vectorizers: list[tuple[str, object]] = [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=self.ngram_range,
                    min_df=self.min_df,
                    max_features=self.max_features,
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words=self.stop_words,
                    sublinear_tf=self.sublinear_tf,
                ),
            )
        ]
        if self.char_ngram_range is not None:
            vectorizers.append(
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=self.char_ngram_range,
                        min_df=self.min_df,
                        max_features=self.char_max_features,
                        lowercase=True,
                        strip_accents="unicode",
                        sublinear_tf=self.sublinear_tf,
                    ),
                )
            )

        self.pipeline = Pipeline(
            steps=[
                ("features", FeatureUnion(transformer_list=vectorizers)),
                ("svd", TruncatedSVD(n_components=self.n_components, random_state=42)),
                ("norm", Normalizer(copy=False)),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.c,
                        class_weight=self.class_weight,
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
        self.pipeline.fit(texts, labels)

    def decision_function(self, texts: list[str]) -> list[float]:
        if self.pipeline is None:
            raise RuntimeError("The model has not been trained or loaded yet.")
        probabilities = self.pipeline.predict_proba(texts)[:, 1]
        return [float(value) for value in probabilities.tolist()]

    def predict(self, texts: list[str]) -> list[int]:
        return [1 if score >= self.threshold else 0 for score in self.decision_function(texts)]

    def save(self, output_dir: str | Path) -> Path:
        if self.pipeline is None:
            raise RuntimeError("Cannot save an untrained model.")
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / "model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(self, handle)
        return model_path

    @classmethod
    def load(cls, model_dir: str | Path) -> "LatentTfidfClassifier":
        model_path = Path(model_dir) / "model.pkl"
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected model type in {model_path}")
        return model


@dataclass(slots=True)
class WeightedEnsembleClassifier:
    member_specs: list[dict[str, object]]
    weights: list[float]
    threshold: float = 0.0
    members: list[object] = field(init=False, default_factory=list, repr=False)

    def fit(self, texts: list[str], labels: list[int]) -> None:
        if not self.member_specs:
            raise ValueError("WeightedEnsembleClassifier requires at least one member spec.")
        if len(self.member_specs) != len(self.weights):
            raise ValueError("member_specs and weights must have the same length.")
        total_weight = sum(float(weight) for weight in self.weights)
        if total_weight <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")

        normalized = [float(weight) / total_weight for weight in self.weights]
        self.weights = normalized
        self.members = []
        for spec in self.member_specs:
            model = _build_ensemble_member(spec)
            model.fit(texts, labels)
            self.members.append(model)

    def decision_function(self, texts: list[str]) -> list[float]:
        if not self.members:
            raise RuntimeError("The ensemble has not been trained or loaded yet.")
        member_scores = [member.decision_function(texts) for member in self.members]
        combined: list[float] = []
        for values in zip(*member_scores):
            combined.append(sum(weight * score for weight, score in zip(self.weights, values, strict=True)))
        return combined

    def predict(self, texts: list[str]) -> list[int]:
        return [1 if score >= self.threshold else 0 for score in self.decision_function(texts)]

    def save(self, output_dir: str | Path) -> Path:
        if not self.members:
            raise RuntimeError("Cannot save an untrained ensemble.")
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / "model.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(self, handle)
        return model_path

    @classmethod
    def load(cls, model_dir: str | Path) -> "WeightedEnsembleClassifier":
        model_path = Path(model_dir) / "model.pkl"
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Unexpected model type in {model_path}")
        return model


def _build_ensemble_member(spec: dict[str, object]) -> object:
    kind = str(spec.get("kind", "tfidf_svm"))
    params = {key: value for key, value in spec.items() if key != "kind"}
    if kind == "tfidf_svm":
        return TfidfSvmClassifier(**params)
    if kind == "latent_tfidf":
        return LatentTfidfClassifier(**params)
    raise ValueError(f"Unsupported ensemble member kind: {kind}")


@dataclass(slots=True)
class ArtifactEnsembleClassifier:
    member_specs: list[dict[str, object]]
    weights: list[float]
    threshold: float = 0.0
    normalization: str = "zscore"
    calibrators: list[dict[str, float]] = field(default_factory=list)
    members: list[object] = field(init=False, default_factory=list, repr=False)

    def fit(self, texts: list[str], labels: list[int]) -> None:
        del labels
        if not self.member_specs:
            raise ValueError("ArtifactEnsembleClassifier requires at least one member spec.")
        if len(self.member_specs) != len(self.weights):
            raise ValueError("member_specs and weights must have the same length.")

        total_weight = sum(float(weight) for weight in self.weights)
        if total_weight <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        self.weights = [float(weight) / total_weight for weight in self.weights]

        self.members = [_load_artifact_member(spec) for spec in self.member_specs]
        if self.calibrators and len(self.calibrators) == len(self.member_specs):
            self.calibrators = [
                {
                    "mean": float(spec.get("mean", 0.0)),
                    "std": max(float(spec.get("std", 1.0)), 1e-12),
                }
                for spec in self.calibrators
            ]
            return

        self.calibrators = []
        for member in self.members:
            scores = [float(value) for value in member.decision_function(texts)]
            mean = sum(scores) / len(scores)
            variance = sum((value - mean) ** 2 for value in scores) / len(scores)
            self.calibrators.append(
                {
                    "mean": mean,
                    "std": max(math.sqrt(variance), 1e-12),
                }
            )

    def decision_function(self, texts: list[str]) -> list[float]:
        if not self.members:
            raise RuntimeError("The artifact ensemble has not been trained or loaded yet.")
        if len(self.calibrators) != len(self.members):
            raise RuntimeError("Artifact ensemble calibration statistics are missing.")

        combined = [0.0] * len(texts)
        for member, weight, calibrator in zip(self.members, self.weights, self.calibrators, strict=True):
            scores = [float(value) for value in member.decision_function(texts)]
            if self.normalization == "zscore":
                mean = float(calibrator["mean"])
                std = max(float(calibrator["std"]), 1e-12)
                scores = [(value - mean) / std for value in scores]
            elif self.normalization != "none":
                raise ValueError(f"Unsupported artifact ensemble normalization: {self.normalization}")

            for index, score in enumerate(scores):
                combined[index] += weight * score
        return combined

    def predict(self, texts: list[str]) -> list[int]:
        return [1 if score >= self.threshold else 0 for score in self.decision_function(texts)]

    def save(self, output_dir: str | Path) -> Path:
        if not self.members:
            raise RuntimeError("Cannot save an uninitialized artifact ensemble.")
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        runtime_path = target_dir / "artifact_ensemble_runtime.json"
        payload = {
            "member_specs": self.member_specs,
            "weights": self.weights,
            "threshold": self.threshold,
            "normalization": self.normalization,
            "calibrators": self.calibrators,
        }
        runtime_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return runtime_path

    @classmethod
    def load(cls, model_dir: str | Path) -> "ArtifactEnsembleClassifier":
        runtime_path = Path(model_dir) / "artifact_ensemble_runtime.json"
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        model = cls(
            member_specs=list(payload["member_specs"]),
            weights=[float(value) for value in payload["weights"]],
            threshold=float(payload.get("threshold", 0.0)),
            normalization=str(payload.get("normalization", "zscore")),
            calibrators=[
                {
                    "mean": float(item.get("mean", 0.0)),
                    "std": max(float(item.get("std", 1.0)), 1e-12),
                }
                for item in payload.get("calibrators", [])
            ],
        )
        model.members = [_load_artifact_member(spec) for spec in model.member_specs]
        return model


def _load_artifact_member(spec: dict[str, object]) -> object:
    kind = str(spec.get("kind", "")).strip()
    model_dir = spec.get("model_dir")
    if not kind:
        raise ValueError("Artifact ensemble member specs require a 'kind'.")
    if not model_dir:
        raise ValueError("Artifact ensemble member specs require a 'model_dir'.")

    root = Path(str(model_dir))
    if kind == "tfidf_svm":
        return TfidfSvmClassifier.load(root)
    if kind == "latent_tfidf":
        return LatentTfidfClassifier.load(root)
    if kind == "tfidf_ensemble":
        return WeightedEnsembleClassifier.load(root)
    if kind == "roberta":
        return RobertaClassifierScaffold.load(root)
    if kind == "roberta_probe":
        return RobertaProbeClassifier.load(root)
    raise ValueError(f"Unsupported artifact ensemble member kind: {kind}")


@dataclass(slots=True)
class RobertaClassifierScaffold:
    pretrained_model_name: str = "roberta-base"
    max_length: int = 256
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    threshold: float = 0.0
    allow_download: bool = False
    cache_dir: str | None = None
    huggingface_token: str | None = None
    freeze_backbone: bool = False
    use_class_weights: bool = True
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    model: object | None = field(init=False, default=None, repr=False)
    tokenizer: object | None = field(init=False, default=None, repr=False)
    backend: str = field(init=False, default="", repr=False)
    local_model: object | None = field(init=False, default=None, repr=False)
    local_vocab: dict[str, int] | None = field(init=False, default=None, repr=False)
    device: str = field(init=False, default="cpu")

    def fit(self, texts: list[str], labels: list[int]) -> None:
        torch, transformers = _require_transformers_stack()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer, self.model = self._load_huggingface_backbone(transformers)
        except Exception:
            self.tokenizer = None
            self.model = None
        else:
            if self.model is None:
                raise RuntimeError("Expected a Hugging Face model instance after loading.")
            if self.freeze_backbone:
                self._freeze_huggingface_backbone()
            self.model.to(self.device)
            self.backend = "transformers"
            self._fit_huggingface(torch, transformers, texts, labels)
            return

        self.backend = "local_tiny"
        self._fit_local_tiny_transformer(torch, texts, labels)

    def predict(self, texts: list[str]) -> list[int]:
        if self.backend == "transformers":
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("The RoBERTa model has not been trained or loaded yet.")
            return [1 if score >= self.threshold else 0 for score in self.decision_function(texts)]
        if self.backend == "local_tiny":
            if self.local_model is None or self.local_vocab is None:
                raise RuntimeError("The local transformer model has not been trained or loaded yet.")
            return [1 if score >= self.threshold else 0 for score in self.decision_function(texts)]
        raise RuntimeError("The RoBERTa model has not been trained or loaded yet.")

    def decision_function(self, texts: list[str]) -> list[float]:
        if self.backend == "transformers":
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("The RoBERTa model has not been trained or loaded yet.")
            torch, _ = _require_transformers_stack()
            scores: list[float] = []
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(texts), self.batch_size):
                    batch_texts = texts[start : start + self.batch_size]
                    encoded = self.tokenizer(
                        batch_texts,
                        truncation=True,
                        padding=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(self.device) for key, value in encoded.items()}
                    logits = self.model(**encoded).logits
                    margins = logits[:, 1] - logits[:, 0]
                    scores.extend(float(value) for value in margins.detach().cpu().tolist())
            return scores

        if self.backend == "local_tiny":
            if self.local_model is None or self.local_vocab is None:
                raise RuntimeError("The local transformer model has not been trained or loaded yet.")
            return self._local_decision_function(texts)

        raise RuntimeError("The RoBERTa model has not been trained or loaded yet.")

    def save(self, output_dir: str | Path) -> Path:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.backend == "transformers":
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Cannot save an untrained RoBERTa model.")
            self.model.save_pretrained(target_dir)
            self.tokenizer.save_pretrained(target_dir)
        elif self.backend == "local_tiny":
            if self.local_model is None or self.local_vocab is None:
                raise RuntimeError("Cannot save an untrained local transformer model.")
            torch, _ = _require_transformers_stack()
            torch.save(self.local_model.state_dict(), target_dir / "local_tiny_model.pt")
            (target_dir / "local_vocab.json").write_text(json.dumps(self.local_vocab, indent=2), encoding="utf-8")
        else:
            raise RuntimeError("The RoBERTa model has not been trained or loaded yet.")
        config_path = target_dir / "roberta_runtime.json"
        payload = {
            "pretrained_model_name": self.pretrained_model_name,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "threshold": self.threshold,
            "allow_download": self.allow_download,
            "cache_dir": self.cache_dir,
            "freeze_backbone": self.freeze_backbone,
            "use_class_weights": self.use_class_weights,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "backend": self.backend,
        }
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return config_path

    @classmethod
    def load(cls, model_dir: str | Path) -> "RobertaClassifierScaffold":
        torch, transformers = _require_transformers_stack()
        root = Path(model_dir)
        config_path = root / "roberta_runtime.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        backend = str(payload.pop("backend", "transformers"))
        model = cls(**payload)
        model.device = "cuda" if torch.cuda.is_available() else "cpu"
        model.backend = backend
        if backend == "transformers":
            model.tokenizer = transformers.AutoTokenizer.from_pretrained(root)
            model.model = transformers.AutoModelForSequenceClassification.from_pretrained(root)
            model.model.to(model.device)
            model.model.eval()
            return model
        if backend == "local_tiny":
            model.local_vocab = json.loads((root / "local_vocab.json").read_text(encoding="utf-8"))
            model.local_model = model._build_local_tiny_model(torch, len(model.local_vocab))
            state_dict = torch.load(root / "local_tiny_model.pt", map_location=model.device)
            model.local_model.load_state_dict(state_dict)
            model.local_model.to(model.device)
            model.local_model.eval()
            return model
        raise RuntimeError(f"Unsupported saved RoBERTa backend: {backend}")
        return model

    def _load_huggingface_backbone(self, transformers):
        load_kwargs = {
            "local_files_only": not self.allow_download,
        }
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir
        if self.huggingface_token:
            load_kwargs["token"] = self.huggingface_token

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.pretrained_model_name,
            **load_kwargs,
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=2,
            **load_kwargs,
        )
        return tokenizer, model

    def _fit_huggingface(self, torch, transformers, texts: list[str], labels: list[int]) -> None:
        assert self.model is not None
        assert self.tokenizer is not None
        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters are available for RoBERTa training.")

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_batches = max((len(texts) + self.batch_size - 1) // self.batch_size, 1)
        total_steps = max(total_batches * self.epochs, 1)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(int(total_steps * self.warmup_ratio), 0),
            num_training_steps=total_steps,
        )

        class_weights = None
        if self.use_class_weights:
            positive_count = sum(labels)
            negative_count = max(len(labels) - positive_count, 1)
            positive_weight = max(negative_count / max(positive_count, 1), 1.0)
            class_weights = torch.tensor([1.0, positive_weight], dtype=torch.float32, device=self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        for epoch in range(self.epochs):
            indices = list(range(len(texts)))
            random.Random(42 + epoch).shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch_texts = [texts[index] for index in batch_indices]
                batch_labels = [labels[index] for index in batch_indices]
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                label_tensor = torch.tensor(batch_labels, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                outputs = self.model(**encoded)
                loss = criterion(outputs.logits, label_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

        self.model.eval()

    def _freeze_huggingface_backbone(self) -> None:
        assert self.model is not None
        backbone = getattr(self.model, "roberta", None)
        if backbone is None:
            return
        for parameter in backbone.parameters():
            parameter.requires_grad = False


@dataclass(slots=True)
class RobertaProbeClassifier:
    pretrained_model_name: str = "roberta-base"
    max_length: int = 128
    batch_size: int = 8
    threshold: float = 0.0
    allow_download: bool = False
    cache_dir: str | None = None
    huggingface_token: str | None = None
    c: float = 1.0
    class_weight: str | dict[int, float] | None = "balanced"
    device: str = field(init=False, default="cpu")
    tokenizer: object | None = field(init=False, default=None, repr=False)
    encoder: object | None = field(init=False, default=None, repr=False)
    classifier: object | None = field(init=False, default=None, repr=False)

    def fit(self, texts: list[str], labels: list[int]) -> None:
        torch, transformers = _require_transformers_stack()
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise RuntimeError("RobertaProbeClassifier requires scikit-learn.") from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.encoder = self._load_encoder(transformers)
        self.encoder.to(self.device)
        self.encoder.eval()

        train_features = self._encode_texts(torch, texts)
        classifier = LogisticRegression(
            C=self.c,
            class_weight=self.class_weight,
            max_iter=2000,
            solver="liblinear",
            random_state=42,
        )
        classifier.fit(train_features, labels)
        self.classifier = classifier

    def predict(self, texts: list[str]) -> list[int]:
        return [1 if score >= self.threshold else 0 for score in self.decision_function(texts)]

    def decision_function(self, texts: list[str]) -> list[float]:
        if self.classifier is None or self.encoder is None or self.tokenizer is None:
            raise RuntimeError("The RoBERTa probe has not been trained or loaded yet.")
        torch, _ = _require_transformers_stack()
        features = self._encode_texts(torch, texts)
        raw_scores = self.classifier.decision_function(features)
        if hasattr(raw_scores, "tolist"):
            return [float(value) for value in raw_scores.tolist()]
        return [float(value) for value in raw_scores]

    def save(self, output_dir: str | Path) -> Path:
        if self.classifier is None:
            raise RuntimeError("Cannot save an untrained RoBERTa probe.")
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        head_path = target_dir / "probe_head.pkl"
        with head_path.open("wb") as handle:
            pickle.dump(self.classifier, handle)
        config_path = target_dir / "roberta_probe_runtime.json"
        payload = {
            "pretrained_model_name": self.pretrained_model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "threshold": self.threshold,
            "allow_download": self.allow_download,
            "cache_dir": self.cache_dir,
            "c": self.c,
            "class_weight": self.class_weight,
        }
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return config_path

    @classmethod
    def load(cls, model_dir: str | Path) -> "RobertaProbeClassifier":
        _, transformers = _require_transformers_stack()
        root = Path(model_dir)
        payload = json.loads((root / "roberta_probe_runtime.json").read_text(encoding="utf-8"))
        model = cls(**payload)
        model.tokenizer, model.encoder = model._load_encoder(transformers)
        with (root / "probe_head.pkl").open("rb") as handle:
            model.classifier = pickle.load(handle)
        return model

    def _load_encoder(self, transformers):
        load_kwargs = {
            "local_files_only": not self.allow_download,
        }
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir
        if self.huggingface_token:
            load_kwargs["token"] = self.huggingface_token

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.pretrained_model_name,
            **load_kwargs,
        )
        encoder = transformers.AutoModel.from_pretrained(
            self.pretrained_model_name,
            **load_kwargs,
        )
        return tokenizer, encoder

    def _encode_texts(self, torch, texts: list[str]) -> list[list[float]]:
        assert self.tokenizer is not None
        assert self.encoder is not None
        features: list[list[float]] = []
        self.encoder.eval()
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.encoder(**encoded)
                pooled = outputs.last_hidden_state[:, 0, :]
                features.extend([list(map(float, row)) for row in pooled.cpu().tolist()])
        return features

    def _fit_local_tiny_transformer(self, torch, texts: list[str], labels: list[int]) -> None:
        self.local_vocab = self._build_local_vocab(texts)
        self.local_model = self._build_local_tiny_model(torch, len(self.local_vocab))
        self.local_model.to(self.device)

        positive_count = sum(labels)
        negative_count = max(len(labels) - positive_count, 1)
        positive_weight = max(negative_count / max(positive_count, 1), 1.0)
        class_weights = torch.tensor([1.0, positive_weight], dtype=torch.float32, device=self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        local_learning_rate = max(self.learning_rate, 3e-4)
        optimizer = torch.optim.AdamW(self.local_model.parameters(), lr=local_learning_rate)

        self.local_model.train()
        for epoch in range(self.epochs):
            indices = list(range(len(texts)))
            random.Random(42 + epoch).shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch_texts = [texts[index] for index in batch_indices]
                batch_labels = [labels[index] for index in batch_indices]
                input_ids, attention_mask = self._encode_local_batch(torch, batch_texts)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label_tensor = torch.tensor(batch_labels, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = self.local_model(input_ids, attention_mask)
                loss = criterion(logits, label_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                optimizer.step()

        self.local_model.eval()

    def _local_decision_function(self, texts: list[str]) -> list[float]:
        torch, _ = _require_transformers_stack()
        assert self.local_model is not None
        scores: list[float] = []
        self.local_model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                input_ids, attention_mask = self._encode_local_batch(torch, batch_texts)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                logits = self.local_model(input_ids, attention_mask)
                margins = logits[:, 1] - logits[:, 0]
                scores.extend(float(value) for value in margins.detach().cpu().tolist())
        return scores

    def _build_local_tiny_model(self, torch, vocab_size: int):
        hidden_size = 96
        num_heads = 4
        num_layers = 2
        max_length = self.max_length

        class TinyTransformerClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embedding = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=0)
                self.position_embedding = torch.nn.Embedding(max_length, hidden_size)
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 2,
                    dropout=0.1,
                    batch_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.dropout = torch.nn.Dropout(0.1)
                self.classifier = torch.nn.Linear(hidden_size, 2)

            def forward(self, input_ids, attention_mask):
                positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
                hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
                hidden = self.encoder(hidden, src_key_padding_mask=(attention_mask == 0))
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = self.dropout(pooled)
                return self.classifier(pooled)

        return TinyTransformerClassifier()

    def _build_local_vocab(self, texts: list[str], max_vocab_size: int = 30000) -> dict[str, int]:
        counts = Counter()
        for text in texts:
            counts.update(token for token in text.lower().split() if token)
        vocab = {"[PAD]": 0, "[UNK]": 1}
        for token, _ in counts.most_common(max_vocab_size - len(vocab)):
            if token not in vocab:
                vocab[token] = len(vocab)
        return vocab

    def _encode_local_batch(self, torch, texts: list[str]):
        assert self.local_vocab is not None
        encoded_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for text in texts:
            token_ids = [
                self.local_vocab.get(token, self.local_vocab["[UNK]"])
                for token in text.lower().split()[: self.max_length]
            ]
            if not token_ids:
                token_ids = [self.local_vocab["[UNK]"]]
            attention = [1] * len(token_ids)
            if len(token_ids) < self.max_length:
                pad_count = self.max_length - len(token_ids)
                token_ids.extend([self.local_vocab["[PAD]"]] * pad_count)
                attention.extend([0] * pad_count)
            encoded_rows.append(token_ids[: self.max_length])
            mask_rows.append(attention[: self.max_length])
        return (
            torch.tensor(encoded_rows, dtype=torch.long),
            torch.tensor(mask_rows, dtype=torch.long),
        )


def _require_transformers_stack():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "RoBERTa training requires the optional transformer stack. "
            "Install compatible versions of `torch` and `transformers` before using model_name='roberta'."
        ) from exc
    return torch, transformers
