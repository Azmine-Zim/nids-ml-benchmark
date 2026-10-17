"""NSL-KDD loader.

The tables are the public KDDTrain+ and KDDTest+ files from the UNB NSL-KDD
release: 41 connection features, a label, and a difficulty score. There is no
header row. This module does not sample the files. If a caller passes a
smaller copy, the run record should say so.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMN_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]
LABEL_COLUMN = "label"
DIFFICULTY_COLUMN = "difficulty"
TRAIN_NAME = "KDDTrain+.txt"
TEST_NAME = "KDDTest+.txt"

# Public copies of the UNB files. The + in the filename is percent-encoded.
MIRRORS = [
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/{name}",
    "https://media.githubusercontent.com/media/jmnwong/NSL-KDD-Dataset/master/{name}",
]


def clean_label(labels: pd.Series) -> pd.Series:
    return labels.astype(str).str.strip().str.lower().str.rstrip(".")


def attack_flag(labels: pd.Series) -> pd.Series:
    """1 for any attack name, 0 for normal."""
    return (clean_label(labels) != "normal").astype(int)


def read_nslkdd_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    if frame.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"{path} has {frame.shape[1]} columns, expected {len(COLUMN_NAMES)}"
        )
    frame[LABEL_COLUMN] = clean_label(frame[LABEL_COLUMN])
    frame["attack"] = (frame[LABEL_COLUMN] != "normal").astype(int)
    return frame


def download_nslkdd(data_dir: Path) -> tuple[Path, Path]:
    """Download KDDTrain+ and KDDTest+ if they are not already on disk."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in (TRAIN_NAME, TEST_NAME):
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            paths.append(dest)
            continue
        _download_one(name, dest)
        paths.append(dest)
    return paths[0], paths[1]


def _download_one(name: str, dest: Path) -> None:
    errors: list[str] = []
    encoded = name.replace("+", "%2B")
    for template in MIRRORS:
        url = template.format(name=encoded)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nidslab"})
            with urllib.request.urlopen(request, timeout=180) as response:
                dest.write_bytes(response.read())
            if dest.stat().st_size == 0:
                raise RuntimeError("empty file")
            return
        except Exception as exc:  # try the next mirror
            errors.append(f"{url}: {exc}")
            if dest.exists():
                dest.unlink()
    raise RuntimeError(
        "Could not download "
        f"{name}. Put {TRAIN_NAME} and {TEST_NAME} in the data folder yourself "
        "(UNB NSL-KDD, or a public mirror of those two files). "
        "Tried: " + " | ".join(errors)
    )


def feature_columns() -> list[str]:
    return [name for name in COLUMN_NAMES if name not in (LABEL_COLUMN, DIFFICULTY_COLUMN)]


def with_attack_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy the frame and add a binary attack column if it is missing."""
    out = frame.copy()
    if "attack" not in out.columns:
        out["attack"] = attack_flag(out[LABEL_COLUMN])
    return out


def preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fit_on: str = "normal",
) -> tuple[object, object, object, object, dict]:
    """Encode features and return matrices for one detector.

    fit_on="normal" fits the encoder on normal training rows and returns only
    those rows as the training matrix. fit_on="all" fits on every training
    row and returns the full training matrix with labels. The test matrix is
    always the full test file.
    """
    if fit_on not in ("normal", "all"):
        raise ValueError("fit_on must be 'normal' or 'all'")
    train = with_attack_column(train)
    test = with_attack_column(test)
    columns = feature_columns()
    numeric = [name for name in columns if name not in CATEGORICAL]
    y_train = train["attack"].to_numpy()
    y_test = test["attack"].to_numpy()
    normal_mask = y_train == 0
    if int(normal_mask.sum()) == 0:
        raise ValueError("training split has no normal rows")

    transformer = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("num", StandardScaler(), numeric),
        ]
    )
    if fit_on == "normal":
        transformer.fit(train.loc[normal_mask, columns])
        x_train = transformer.transform(train.loc[normal_mask, columns])
        y_fit = np.zeros(x_train.shape[0], dtype=int)
    else:
        transformer.fit(train[columns])
        x_train = transformer.transform(train[columns])
        y_fit = y_train.astype(int)
    x_test = transformer.transform(test[columns])
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
        "sampled": False,
    }
    return x_train, y_fit, x_test, y_test, summary


def preprocess_one_class(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[object, object, object, dict]:
    """Fit scaling and one-hot encoding on normal training rows only."""
    x_train, _y_fit, x_test, y_test, summary = preprocess(train, test, fit_on="normal")
    return x_train, x_test, y_test, summary
