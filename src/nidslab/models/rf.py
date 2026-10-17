"""Random Forest trained with labels. This is the supervised reference."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestDetector:
    def __init__(self, seed: int = 42, n_estimators: int = 100) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestDetector":
        self.model.fit(x, y)
        return self

    def anomaly_scores(self, x: np.ndarray) -> np.ndarray:
        """Probability of the attack class. Higher means more attack-like."""
        return self.model.predict_proba(x)[:, 1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.anomaly_scores(x) >= 0.5).astype(int)
