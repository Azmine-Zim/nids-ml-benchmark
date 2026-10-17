"""CICIDS2017 loader.

The files are the eight public MachineLearningCSV tables from the Canadian
Institute for Cybersecurity (Sharafaldin, Habibi Lashkari, and Ghorbani, 2018).

Those tables together have on the order of 2.8 million flows. This loader does
not train on every row. It streams every file, drops rows with missing or
infinite values, and keeps a reservoir sample of BENIGN and attack flows.
The sample is then split 80/20. That split is ours. CICIDS2017 has no official
train and test files.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nidslab.data.tabular import preprocess_numeric

CSV_NAMES = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
)

# Public mirror of the CIC MachineLearningCSV.zip (about 224 MB).
# The older UNB direct links now redirect to an HTML page.
ZIP_URL = "https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/csvs/MachineLearningCSV.zip"
ZIP_NAME = "MachineLearningCSV.zip"

BENIGN_CAP = 20_000
ATTACK_CAP = 20_000
TEST_SIZE = 0.2

CITATION = (
    "Sharafaldin, I., Habibi Lashkari, A., and Ghorbani, A. A. (2018). "
    "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. "
    "ICISSP."
)

def sample_note(seed: int, benign_cap: int, attack_cap: int) -> str:
    return (
        "Not the full CICIDS2017 table. Every MachineLearningCSV file is streamed. "
        f"A reservoir sample keeps at most {benign_cap} BENIGN rows and {attack_cap} attack rows "
        f"(seed {seed}). Infinite or missing values are dropped before sampling. "
        f"The sample is split {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} "
        "into train and test, stratified on the attack bit. "
        "CICIDS2017 has no official train/test files, so this split is defined here. "
        "The caps are equal, so the sample is balanced. The raw files are not: "
        "most flows are BENIGN. Metrics are for this balanced sample, not the original mix. "
        "Train and test are a random split of the same days, so a supervised model can look very strong. "
        "Destination Port stays in the features as a number."
    )


def download_cicids(data_dir: Path) -> Path:
    """Download and extract MachineLearningCSV.zip if the eight CSVs are missing."""
    data_dir = Path(data_dir)
    folder = data_dir / "cicids2017"
    folder.mkdir(parents=True, exist_ok=True)
    if _csvs_ready(folder):
        return folder
    zip_path = folder / ZIP_NAME
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        _download(ZIP_URL, zip_path)
    _extract(zip_path, folder)
    if not _csvs_ready(folder):
        found = sorted(path.name for path in folder.rglob("*.csv"))
        raise RuntimeError(
            "CICIDS2017 CSVs were not found after extract. "
            f"Looked for {', '.join(CSV_NAMES)}. Found: {found}. "
            "You can unzip MachineLearningCSV.zip into data/cicids2017 yourself."
        )
    return folder


def load_cicids_splits(
    data_dir: Path,
    seed: int = 42,
    benign_cap: int = BENIGN_CAP,
    attack_cap: int = ATTACK_CAP,
    download: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    folder = Path(data_dir) / "cicids2017"
    if download and not _csvs_ready(folder):
        download_cicids(data_dir)
    paths = _csv_paths(folder)
    train, test, summary = sample_cicids_frames(
        paths,
        seed=seed,
        benign_cap=benign_cap,
        attack_cap=attack_cap,
    )
    return train, test, summary


def sample_cicids_frames(
    paths: list[Path],
    seed: int = 42,
    benign_cap: int = BENIGN_CAP,
    attack_cap: int = ATTACK_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    feature_cols = _shared_feature_columns(paths)
    rng = np.random.default_rng(seed)
    benign: list | np.ndarray = []
    attack: list | np.ndarray = []
    benign_seen = 0
    attack_seen = 0
    dropped = 0
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
            values, labels, n_dropped = _clean_chunk(chunk, feature_cols)
            dropped += n_dropped
            if len(values) == 0:
                continue
            benign_rows = values[labels == 0]
            attack_rows = values[labels == 1]
            benign, benign_seen = _update_reservoir(benign, benign_seen, benign_rows, benign_cap, rng)
            attack, attack_seen = _update_reservoir(attack, attack_seen, attack_rows, attack_cap, rng)
    benign_arr = _as_array(benign, len(feature_cols))
    attack_arr = _as_array(attack, len(feature_cols))
    if len(benign_arr) < 2 or len(attack_arr) < 2:
        raise RuntimeError("CICIDS sample is too small after dropping bad rows.")
    frame = pd.DataFrame(
        np.vstack([benign_arr, attack_arr]),
        columns=feature_cols,
    )
    frame["attack"] = np.array([0] * len(benign_arr) + [1] * len(attack_arr))
    train, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=frame["attack"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.attrs["sampled"] = True
    test.attrs["sampled"] = True
    summary = {
        "source": "CICIDS2017 MachineLearningCSV",
        "citation": CITATION,
        "files": [path.name for path in paths],
        "rows_seen_benign": int(benign_seen),
        "rows_seen_attack": int(attack_seen),
        "rows_dropped_nonfinite": int(dropped),
        "benign_cap": int(benign_cap),
        "attack_cap": int(attack_cap),
        "sample_benign": int(len(benign_arr)),
        "sample_attack": int(len(attack_arr)),
        "sampled": True,
        "sample_note": sample_note(seed, benign_cap, attack_cap),
        "split_train_label": "train sample (80%)",
        "split_test_label": "test sample (20%)",
    }
    return train, test, summary


def prepare_cicids(train: pd.DataFrame, test: pd.DataFrame, fit_on: str = "normal"):
    x_train, y_fit, x_test, y_test, summary = preprocess_numeric(train, test, fit_on=fit_on)
    summary["sampled"] = True
    summary["split_train_label"] = "train sample (80%)"
    summary["split_test_label"] = "test sample (20%)"
    summary["sample_note"] = sample_note(42, BENIGN_CAP, ATTACK_CAP)
    return x_train, y_fit, x_test, y_test, summary


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "nidslab"})
    with urllib.request.urlopen(request, timeout=120) as response:
        magic = response.read(4)
        if magic != b"PK\x03\x04":
            raise RuntimeError(f"Download from {url} was not a zip file.")
        with dest.open("wb") as handle:
            handle.write(magic)
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)


def _extract(zip_path: Path, folder: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(folder)


def _csvs_ready(folder: Path) -> bool:
    try:
        _csv_paths(folder)
    except FileNotFoundError:
        return False
    return True


def _csv_paths(folder: Path) -> list[Path]:
    found = {path.name: path for path in folder.rglob("*.csv")}
    missing = [name for name in CSV_NAMES if name not in found]
    if missing:
        raise FileNotFoundError("missing CICIDS files: " + ", ".join(missing))
    return [found[name] for name in CSV_NAMES]


def _shared_feature_columns(paths: list[Path]) -> list[str]:
    shared: set[str] | None = None
    ordered: list[str] | None = None
    for path in paths:
        header = pd.read_csv(path, nrows=0)
        names = [str(column).strip() for column in header.columns]
        features = [name for name in names if name.lower() != "label"]
        if "Label" not in names and "label" not in [name.lower() for name in names]:
            raise ValueError(f"{path.name} has no Label column")
        if shared is None:
            shared = set(features)
            ordered = features
        else:
            shared &= set(features)
    columns = [name for name in ordered or [] if name in (shared or set())]
    if not columns:
        raise ValueError("CICIDS files have no shared numeric columns")
    return columns


def _clean_chunk(chunk: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    chunk = chunk.copy()
    chunk.columns = [str(column).strip() for column in chunk.columns]
    labels = chunk["Label"].astype(str).str.strip()
    usable = labels.ne("") & ~labels.str.lower().isin(["nan", "none"])
    numbers = chunk.loc[usable, feature_cols].apply(pd.to_numeric, errors="coerce")
    numbers = numbers.replace([np.inf, -np.inf], np.nan)
    finite = numbers.notna().all(axis=1)
    dropped = int(len(chunk) - int(finite.sum()))
    kept = numbers.loc[finite]
    kept_labels = labels.loc[usable].loc[finite]
    attack = (kept_labels.str.lower() != "benign").to_numpy().astype(int)
    return kept.to_numpy(dtype=np.float64), attack, dropped


def _update_reservoir(store, seen: int, rows: np.ndarray, capacity: int, rng: np.random.Generator):
    if len(rows) == 0 or capacity <= 0:
        return store, seen
    if seen < capacity:
        need = capacity - seen
        take = min(need, len(rows))
        block = rows[:take]
        if isinstance(store, list):
            store.extend(block.tolist())
        else:
            store = np.vstack([store, block])
        seen += take
        rows = rows[take:]
    if len(rows) == 0:
        return store, seen
    if isinstance(store, list):
        store = np.asarray(store, dtype=np.float64)
    positions = np.arange(seen + 1, seen + len(rows) + 1)
    chosen = rng.integers(0, positions)
    for index in np.flatnonzero(chosen < capacity):
        store[int(chosen[index])] = rows[index]
    return store, seen + len(rows)


def _as_array(store, width: int) -> np.ndarray:
    if isinstance(store, list):
        if not store:
            return np.empty((0, width), dtype=np.float64)
        return np.asarray(store, dtype=np.float64)
    return np.asarray(store, dtype=np.float64)
