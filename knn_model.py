"""An explainable KNN recurrence classifier, implemented from scratch."""
from __future__ import annotations
import math
import numpy as np


class KNNRecurrenceModel:
    def __init__(self, k: int = 5):
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = int(k)
        self.feature_names_: list[str] = []
        self.mean_ = None
        self.scale_ = None
        self.matrix_ = None
        self.labels_ = None

    def fit(self, features: list[dict[str, float]], labels: list[int]):
        if not labels or len(features) != len(labels) or not set(labels).issubset({0, 1}):
            raise ValueError("KNN needs equally sized non-empty features and binary labels.")
        if len(set(labels)) < 2:
            raise ValueError("KNN needs both repeated and absent examples.")
        self.feature_names_ = sorted(features[0])
        if not self.feature_names_ or any(set(r) != set(self.feature_names_) for r in features):
            raise ValueError("Every KNN row must contain the same numerical features.")
        matrix = np.asarray([[float(r[f]) for f in self.feature_names_] for r in features], dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("KNN features must be finite numbers.")
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        self.matrix_ = (matrix - self.mean_) / self.scale_
        self.labels_ = np.asarray(labels, dtype=int)
        return self

    def nearest_neighbors(self, row: dict[str, float]):
        if self.matrix_ is None:
            raise ValueError("Fit KNN before predicting.")
        if set(row) != set(self.feature_names_):
            raise ValueError("KNN prediction features must match training.")
        vector = np.asarray([float(row[f]) for f in self.feature_names_], dtype=float)
        if not np.isfinite(vector).all():
            raise ValueError("Prediction features must be finite.")
        scaled = (vector - self.mean_) / self.scale_
        distances = np.sqrt(np.sum((self.matrix_ - scaled) ** 2, axis=1))
        indices = np.argsort(distances, kind="stable")[:min(self.k, len(distances))]
        return [{"index": int(i), "distance": float(distances[i]), "label": int(self.labels_[i])} for i in indices]

    def predict_probability(self, row):
        neighbors = self.nearest_neighbors(row)
        return sum(n["label"] for n in neighbors) / len(neighbors)

    def predict_probabilities(self, rows):
        return [self.predict_probability(row) for row in rows]

    def predict(self, rows):
        return [int(p >= .5) for p in self.predict_probabilities(rows)]

    def to_dict(self):
        if self.matrix_ is None:
            raise ValueError("Fit KNN before saving.")
        return {"algorithm": "KNN", "k": self.k, "features": self.feature_names_,
                "mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
                "scaled_training_rows": self.matrix_.tolist(), "labels": self.labels_.tolist()}

    @classmethod
    def from_dict(cls, data):
        model = cls(data["k"])
        model.feature_names_ = list(data["features"])
        model.mean_ = np.asarray(data["mean"], dtype=float)
        model.scale_ = np.asarray(data["scale"], dtype=float)
        model.matrix_ = np.asarray(data["scaled_training_rows"], dtype=float)
        model.labels_ = np.asarray(data["labels"], dtype=int)
        if model.matrix_.ndim != 2 or model.matrix_.shape != (len(model.labels_), len(model.feature_names_)):
            raise ValueError("Invalid saved KNN matrix.")
        if len(model.mean_) != len(model.feature_names_) or len(model.scale_) != len(model.feature_names_):
            raise ValueError("Invalid KNN feature scaling.")
        if not all(np.isfinite(x).all() for x in [model.mean_, model.scale_, model.matrix_]) or np.any(model.scale_ <= 0):
            raise ValueError("Invalid KNN saved numerical values.")
        return model
