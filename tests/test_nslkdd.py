from pathlib import Path

import pandas as pd

from nidslab.data.nslkdd import COLUMN_NAMES, attack_flag, read_nslkdd_file


def test_attack_flag_treats_only_normal_as_negative():
    labels = pd.Series(["normal", "neptune", "Normal.", " smurf "])
    flags = attack_flag(labels).tolist()
    assert flags == [0, 1, 0, 1]


def test_read_nslkdd_file_names_columns(tmp_path: Path):
    row = ["0", "tcp", "http", "SF"] + ["0"] * 37 + ["neptune", "21"]
    assert len(row) == len(COLUMN_NAMES)
    path = tmp_path / "tiny.txt"
    path.write_text(",".join(row) + "\n", encoding="utf-8")
    frame = read_nslkdd_file(path)
    assert list(frame.columns) == COLUMN_NAMES + ["attack"]
    assert int(frame.loc[0, "attack"]) == 1
    assert frame.loc[0, "protocol_type"] == "tcp"
