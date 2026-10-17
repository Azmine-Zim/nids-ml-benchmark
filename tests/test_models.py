import numpy as np
import pandas as pd

from nidslab.data.nslkdd import CATEGORICAL, COLUMN_NAMES
from nidslab.models.autoencoder import AutoencoderDetector
from nidslab.models.ocsvm import OneClassSVMDetector
from nidslab.pipeline import run_model_on_frames


def _row(protocol: str, service: str, flag: str, src_bytes: int, label: str) -> dict:
    values = {name: 0 for name in COLUMN_NAMES if name not in CATEGORICAL + ["label", "difficulty"]}
    values["protocol_type"] = protocol
    values["service"] = service
    values["flag"] = flag
    values["src_bytes"] = src_bytes
    values["label"] = label
    values["difficulty"] = 0
    return values


def _frames():
    train_rows = [_row("tcp", "http", "SF", 80 + i, "normal") for i in range(80)]
    train_rows += [_row("udp", "private", "S0", 40000 + i, "neptune") for i in range(40)]
    test_rows = [_row("tcp", "http", "SF", 90 + i, "normal") for i in range(20)]
    test_rows += [_row("udp", "private", "S0", 42000 + i, "neptune") for i in range(20)]
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def test_ocsvm_samples_and_scores_far_points_higher():
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 0.2, size=(50, 3))
    attacks = rng.normal(5.0, 0.2, size=(15, 3))
    detector = OneClassSVMDetector(seed=0, nu=0.05, max_train_rows=30)
    detector.fit(normal)
    assert detector.used_rows == 30
    scores = detector.anomaly_scores(np.vstack([normal[:15], attacks]))
    assert scores[15:].mean() > scores[:15].mean()


def test_random_forest_on_tiny_table():
    train, test = _frames()
    record, _y, _scores = run_model_on_frames("rf", train, test, seed=0)
    assert record["counts"]["fit_on"] == "all"
    assert record["metrics"]["roc_auc"] > 0.9


def test_autoencoder_error_is_higher_off_the_training_blob():
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 0.1, size=(120, 6))
    attacks = rng.normal(4.0, 0.1, size=(30, 6))
    detector = AutoencoderDetector(seed=0, epochs=8, batch_size=32)
    detector.fit(normal)
    scores = detector.anomaly_scores(np.vstack([normal[:30], attacks]))
    assert scores[30:].mean() > scores[:30].mean()
