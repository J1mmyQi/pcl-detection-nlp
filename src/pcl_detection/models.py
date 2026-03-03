"""Part 3: Baseline model and proposed-approach model definitions."""

from __future__ import annotations

from collections import Counter
import json
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


class TextClassifier(Protocol):
    def fit(self, texts: list[str], labels: list[int]) -> None:
        """Train the classifier."""

    def predict(self, texts: list[str]) -> list[int]:
        """Predict binary labels."""

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
    backend: str | None = field(init=False, default=None)
    pipeline: object | None = field(init=False, default=None, repr=False)
    token_scores: dict[str, float] | None = field(init=False, default=None, repr=False)
    bias: float = field(init=False, default=0.0, repr=False)

    def fit(self, texts: list[str], labels: list[int]) -> None:
        try:
            from sklearn.pipeline import FeatureUnion
            from sklearn.feature_extraction.text import TfidfVectorizer
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

        self.pipeline = Pipeline(
            steps=[
                (
                    "features",
                    FeatureUnion(transformer_list=vectorizers),
                ),
                (
                    "classifier",
                    LinearSVC(
                        C=self.c,
                        class_weight=self.class_weight,
                        random_state=42,
                    ),
                ),
            ]
        )
        self.pipeline.fit(texts, labels)
        self.backend = "sklearn"

    def predict(self, texts: list[str]) -> list[int]:
        if self.backend == "sklearn" and self.pipeline is not None:
            return [int(value) for value in self.pipeline.predict(texts).tolist()]
        if self.backend == "fallback" and self.token_scores is not None:
            return [self._predict_fallback(text) for text in texts]
        if self.pipeline is None and self.token_scores is None:
            raise RuntimeError("The model has not been trained or loaded yet.")
        raise RuntimeError("Model state is inconsistent.")

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
class RobertaClassifierScaffold:
    pretrained_model_name: str = "roberta-base"
    max_length: int = 256
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3

    def fit(self, texts: list[str], labels: list[int]) -> None:
        raise NotImplementedError(
            "RoBERTa fine-tuning is not implemented in this scaffold yet. "
            "Use the tfidf_svm baseline first, then fill in this adapter with transformers/torch."
        )

    def predict(self, texts: list[str]) -> list[int]:
        raise NotImplementedError("RoBERTa inference is not implemented in this scaffold yet.")

    def save(self, output_dir: str | Path) -> Path:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        config_path = target_dir / "roberta_stub.json"
        config_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return config_path

    @classmethod
    def load(cls, model_dir: str | Path) -> "RobertaClassifierScaffold":
        config_path = Path(model_dir) / "roberta_stub.json"
        return cls(**json.loads(config_path.read_text(encoding="utf-8")))
