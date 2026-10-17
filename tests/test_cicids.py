from pathlib import Path

import numpy as np
import pandas as pd

from nidslab.data.cicids import sample_cicids_frames
from nidslab.pipeline import run_model_on_frames


def _write_csv(path: Path, rows: list[tuple[float, float, str]]) -> None:
    lines = ["Destination Port, Flow Duration, Label"]
    for port, duration, label in rows:
        lines.append(f"{port},{duration},{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sample_drops_infinite_rows_and_keeps_both_classes(tmp_path: Path):
    benign = [(80, 10 + i, "BENIGN") for i in range(30)]
    benign.append((80, np.inf, "BENIGN"))
    attacks = [(80, 5000 + i, "DoS Hulk") for i in range(12)]
    _write_csv(tmp_path / "Monday-WorkingHours.pcap_ISCX.csv", benign + attacks)
    train, test, summary = sample_cicids_frames(
        [tmp_path / "Monday-WorkingHours.pcap_ISCX.csv"],
        seed=1,
        benign_cap=20,
        attack_cap=8,
    )
    assert summary["rows_dropped_nonfinite"] == 1
    assert summary["rows_seen_benign"] == 30
    assert summary["sample_benign"] == 20
    assert summary["sample_attack"] == 8
    assert summary["sampled"] is True
    assert set(train["attack"]).issubset({0, 1})
    assert len(train) + len(test) == 28


def test_cicids_frames_can_train_isolation_forest(tmp_path: Path):
    benign = [(80, 10 + (i % 5), "BENIGN") for i in range(40)]
    attacks = [(444, 8000 + i, "PortScan") for i in range(20)]
    _write_csv(tmp_path / "day.csv", benign + attacks)
    train, test, _summary = sample_cicids_frames(
        [tmp_path / "day.csv"],
        seed=2,
        benign_cap=40,
        attack_cap=20,
    )
    from nidslab.data.cicids import prepare_cicids

    record, _y, _scores = run_model_on_frames(
        "iforest",
        train,
        test,
        seed=2,
        prepare=prepare_cicids,
        dataset="cicids2017",
    )
    assert record["dataset"] == "cicids2017"
    assert record["metrics"]["roc_auc"] > 0.7
