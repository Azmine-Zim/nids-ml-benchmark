"""One-Class SVM on a documented sample of normal traffic.

The RBF kernel keeps a row for every training point. The full NSL-KDD normal
set is too large for that, so the caller passes a sample and records the size.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM


class OneClassSVMDetector:
    def __init__(self, seed: int = 42, nu: float = 0.01, max_train_rows: int = 4000) -> None:
        self.seed = seed
        self.nu = nu
        self.max_train_rows = max_train_rows
        self.used_rows = 0
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)

    def fit(self, x_normal: np.ndarray) -> "OneClassSVMDetector":
        x_normal = np.asarray(x_normal)
        if len(x_normal) > self.max_train_rows:
            rng = np.random.default_rng(self.seed)
            picked = rng.choice(len(x_normal), size=self.max_train_rows, replace=False)
            x_fit = x_normal[picked]
        else:
            x_fit = x_normal
        self.used_rows = int(len(x_fit))
        self.model.fit(x_fit)
        return self

    def anomaly_scores(self, x: np.ndarray) -> np.ndarray:
        """Higher means more anomalous. sklearn's decision_function is the other way."""
        return -self.model.decision_function(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        raw = self.model.predict(x)
        return (raw == -1).astype(int)
