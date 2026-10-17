import numpy as np
import pandas as pd

from nidslab.data.nslkdd import CATEGORICAL, COLUMN_NAMES
from nidslab.models.iforest import IsolationForestDetector
from nidslab.pipeline import run_iforest_on_frames


def _row(protocol: str, service: str, flag: str, src_bytes: int, label: str) -> dict:
    values = {name: 0 for name in COLUMN_NAMES if name not in CATEGORICAL + ["label", "difficulty"]}
    values["protocol_type"] = protocol
    values["service"] = service
    values["flag"] = flag
    values["src_bytes"] = src_bytes
    values["label"] = label
    values["difficulty"] = 0
    return values


def test_iforest_scores_far_points_higher():
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0.0, scale=0.3, size=(200, 4))
    attacks = rng.normal(loc=6.0, scale=0.3, size=(40, 4))
    detector = IsolationForestDetector(seed=0, contamination=0.05)
    detector.fit(normal)
    scores = detector.anomaly_scores(np.vstack([normal[:40], attacks]))
    assert scores[40:].mean() > scores[:40].mean()


def test_pipeline_on_tiny_table_separates_attacks():
    # Normal rows vary a little. A constant matrix gives Isolation Forest nothing to split.
    train_rows = [_row("tcp", "http", "SF", 80 + i, "normal") for i in range(80)]
    train_rows += [_row("udp", "private", "S0", 50000, "neptune") for _ in range(10)]
    test_rows = [_row("tcp", "http", "SF", 90 + i, "normal") for i in range(20)]
    test_rows += [_row("udp", "private", "S0", 48000 + i, "neptune") for i in range(20)]
    record = run_iforest_on_frames(
        pd.DataFrame(train_rows), pd.DataFrame(test_rows), seed=0, sampled=True
    )
    assert record["counts"]["train_normal"] == 80
    assert record["counts"]["sampled"] is True
    assert record["metrics"]["roc_auc"] > 0.9
    assert record["metrics"]["tp"] + record["metrics"]["fn"] == 20
