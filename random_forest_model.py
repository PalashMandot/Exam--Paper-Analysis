"""Random Forest recurrence classifier using historical numeric features."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier


class RandomForestRecurrenceModel:
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_: RandomForestClassifier | None = None
        self.feature_names_: list[str] = []

    def fit(self, features: list[dict[str, float]], labels: list[int]):
        if len(features) != len(labels) or not labels:
            raise ValueError("Random Forest requires equally sized, non-empty feature and label lists.")
        if len(set(labels)) < 2:
            raise ValueError("Random Forest needs both recurrence classes in its training history.")
        self.feature_names_ = sorted({key for row in features for key in row})
        matrix = [[float(row.get(name, 0.0)) for name in self.feature_names_] for row in features]
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
            n_jobs=1,
        )
        self.model_.fit(matrix, labels)
        return self

    def predict_probability(self, row: dict[str, float]) -> float:
        if self.model_ is None:
            raise ValueError("Fit the Random Forest model before prediction.")
        matrix = [[float(row.get(name, 0.0)) for name in self.feature_names_]]
        classes = list(self.model_.classes_)
        if 1 not in classes:
            return 0.0
        return float(self.model_.predict_proba(matrix)[0][classes.index(1)])

    def predict(self, rows: list[dict[str, float]]) -> list[int]:
        return [int(self.predict_probability(row) >= 0.5) for row in rows]

    def predict_probabilities(self, rows: list[dict[str, float]]) -> list[float]:
        return [self.predict_probability(row) for row in rows]

    def metadata(self) -> dict[str, Any]:
        return {"algorithm": "Random Forest", "n_estimators": self.n_estimators,
                "random_state": self.random_state, "features": self.feature_names_}
