"""A small, genuine ID3 decision tree for question recurrence features."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _entropy(labels: list[int]) -> float:
    total = len(labels)
    if not total:
        return 0.0
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count)


def _majority_leaf(labels: list[int]) -> dict[str, Any]:
    counts = Counter(labels)
    total = len(labels)
    positive_probability = counts.get(1, 0) / total if total else 0.0
    prediction = 1 if positive_probability >= 0.5 else 0
    return {"leaf": True, "prediction": prediction, "probability": positive_probability}


class ID3Classifier:
    """ID3 trained on categorical features represented as dictionaries."""

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_: dict[str, Any] | None = None
        self.features_: list[str] = []

    def fit(self, features: list[dict[str, Any]], labels: list[int]):
        if len(features) != len(labels) or not labels:
            raise ValueError("ID3 requires equally sized, non-empty feature and label lists.")
        if not set(labels).issubset({0, 1}):
            raise ValueError("ID3 labels must be 0 or 1.")
        self.features_ = sorted({key for row in features for key in row})
        self.tree_ = self._grow(features, labels, self.features_, depth=0)
        return self

    def _grow(self, rows, labels, available, depth):
        if len(set(labels)) == 1 or not available or depth >= self.max_depth:
            return _majority_leaf(labels)
        if len(labels) < 2 * self.min_samples_leaf:
            return _majority_leaf(labels)

        parent_entropy = _entropy(labels)
        best_feature, best_gain = None, 0.0
        for feature in available:
            partitions = {}
            for row, label in zip(rows, labels):
                partitions.setdefault(str(row.get(feature, "")), []).append(label)
            if any(len(part) < self.min_samples_leaf for part in partitions.values()):
                continue
            remainder = sum(len(part) / len(labels) * _entropy(part) for part in partitions.values())
            gain = parent_entropy - remainder
            if gain > best_gain:
                best_feature, best_gain = feature, gain
        if best_feature is None or best_gain <= 1e-12:
            return _majority_leaf(labels)

        node = {"leaf": False, "feature": best_feature, "default": _majority_leaf(labels), "children": {}}
        partitions = {}
        for row, label in zip(rows, labels):
            partitions.setdefault(str(row.get(best_feature, "")), ([], []))
            partitions[str(row.get(best_feature, ""))][0].append(row)
            partitions[str(row.get(best_feature, ""))][1].append(label)
        remaining = [name for name in available if name != best_feature]
        for value, (subset_rows, subset_labels) in partitions.items():
            node["children"][value] = self._grow(subset_rows, subset_labels, remaining, depth + 1)
        return node

    def predict_probability(self, row: dict[str, Any]) -> float:
        if self.tree_ is None:
            raise ValueError("Fit the ID3 model before prediction.")
        node = self.tree_
        while not node["leaf"]:
            value = str(row.get(node["feature"], ""))
            node = node["children"].get(value, node["default"])
        return float(node["probability"])

    def predict(self, rows: list[dict[str, Any]]) -> list[int]:
        return [int(self.predict_probability(row) >= 0.5) for row in rows]

    def predict_probabilities(self, rows: list[dict[str, Any]]) -> list[float]:
        return [self.predict_probability(row) for row in rows]

    def to_dict(self) -> dict[str, Any]:
        if self.tree_ is None:
            raise ValueError("Fit the ID3 model before serialization.")
        return {"algorithm": "ID3", "max_depth": self.max_depth, "min_samples_leaf": self.min_samples_leaf,
                "features": self.features_, "tree": self.tree_}

    @classmethod
    def from_dict(cls, data):
        if data.get("algorithm") != "ID3":
            raise ValueError("This is not an ID3 artifact.")
        model = cls(data.get("max_depth", 8), data.get("min_samples_leaf", 1))
        model.features_ = list(data["features"])
        model.tree_ = data["tree"]
        return model

    def decision_path(self, row):
        if self.tree_ is None:
            raise ValueError("Fit ID3 before explaining a prediction.")
        node, steps = self.tree_, []
        while not node["leaf"]:
            feature = node["feature"]
            value = str(row.get(feature, ""))
            steps.append(f"{feature} = {value}")
            if value not in node["children"]:
                steps.append("Unseen category: use this node's training distribution")
            node = node["children"].get(value, node["default"])
        return steps


def evaluate_predictions(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix": {"true_negative": tn, "false_positive": fp,
                                 "false_negative": fn, "true_positive": tp}}
