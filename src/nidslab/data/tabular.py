"""Numeric preprocessing shared by datasets that are already tabular."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_numeric(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fit_on: str = "normal",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Scale numeric columns. The frames must already contain an attack column."""
    if fit_on not in ("normal", "all"):
        raise ValueError("fit_on must be 'normal' or 'all'")
    feature_cols = [name for name in train.columns if name != "attack"]
    y_train = train["attack"].to_numpy().astype(int)
    y_test = test["attack"].to_numpy().astype(int)
    normal_mask = y_train == 0
    if int(normal_mask.sum()) == 0:
        raise ValueError("training split has no normal rows")

    scaler = StandardScaler()
    if fit_on == "normal":
        scaler.fit(train.loc[normal_mask, feature_cols])
        x_train = scaler.transform(train.loc[normal_mask, feature_cols])
        y_fit = np.zeros(x_train.shape[0], dtype=int)
    else:
        scaler.fit(train[feature_cols])
        x_train = scaler.transform(train[feature_cols])
        y_fit = y_train
    x_test = scaler.transform(test[feature_cols])
    summary = {
        "train_rows": int(len(train)),
        "train_normal": int(normal_mask.sum()),
        "train_attack": int((~normal_mask).sum()),
        "fit_on": fit_on,
        "fit_rows": int(x_train.shape[0]),
        "test_rows": int(len(test)),
        "test_normal": int((y_test == 0).sum()),
        "test_attack": int((y_test == 1).sum()),
        "n_features_after_encoding": int(x_train.shape[1]),
        "sampled": bool(train.attrs.get("sampled", False)),
        "split_train_label": "train sample",
        "split_test_label": "test sample",
    }
    return x_train, y_fit, x_test, y_test, summary
