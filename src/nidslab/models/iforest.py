"""Isolation Forest, trained as a one-class detector on normal traffic."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 100,
        contamination: float = 0.01,
    ) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, x_normal: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(x_normal)
        return self

    def anomaly_scores(self, x: np.ndarray) -> np.ndarray:
        """Higher means more anomalous. sklearn's own score is the other way around."""
        return -self.model.score_samples(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """1 = attack (outlier), 0 = normal (inlier), using sklearn's threshold."""
        raw = self.model.predict(x)
        return (raw == -1).astype(int)
