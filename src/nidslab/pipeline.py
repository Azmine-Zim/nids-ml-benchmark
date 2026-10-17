"""Train one detector, or all four, on NSL-KDD and write metrics plus figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nidslab.data.cicids import load_cicids_splits, prepare_cicids
from nidslab.data.nslkdd import (
    TEST_NAME,
    TRAIN_NAME,
    download_nslkdd,
    preprocess,
    preprocess_one_class,
    read_nslkdd_file,
)
from nidslab.eval.metrics import evaluate_binary
from nidslab.eval.plots import save_confusion, save_roc, save_roc_overlay
from nidslab.models.autoencoder import AutoencoderDetector
from nidslab.models.iforest import IsolationForestDetector
from nidslab.models.ocsvm import OneClassSVMDetector
from nidslab.models.rf import RandomForestDetector

MODEL_NAMES = ("iforest", "ocsvm", "rf", "ae")

PROTOCOLS = {
    "iforest": (
        "One-class. Isolation Forest is fit only on normal rows from KDDTrain+. "
        "Threshold is sklearn's contamination offset (contamination=0.01, 100 trees)."
    ),
    "ocsvm": (
        "One-class. One-Class SVM (RBF, nu=0.01) is fit on a random sample of "
        "4000 normal KDDTrain+ rows. The full normal set is too large for an RBF kernel. "
        "The test file is not sampled."
    ),
    "rf": (
        "Supervised reference. Random Forest (100 trees) is fit on all KDDTrain+ rows "
        "with the attack label. A score of 0.5 is the decision threshold. "
        "This is not comparable one-to-one with the one-class models."
    ),
    "ae": (
        "One-class. A plain dense autoencoder (64-32-16-32-64, MSE loss, no KL term) "
        "is fit on 90% of the normal KDDTrain+ rows for 15 epochs. "
        "The anomaly score is mean squared reconstruction error. "
        "The threshold is the 99th percentile of that error on the held-out 10% of normal rows."
    ),
}


def run_iforest_on_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int = 42,
    sampled: bool = False,
) -> dict:
    x_normal, x_test, y_test, summary = preprocess_one_class(train, test)
    summary["sampled"] = sampled
    detector = IsolationForestDetector(seed=seed)
    detector.fit(x_normal)
    scores = detector.anomaly_scores(x_test)
    predictions = detector.predict(x_test)
    metrics = evaluate_binary(y_test, predictions, scores)
    return _record("iforest", seed, summary, metrics, extra={
        "n_estimators": detector.n_estimators,
        "contamination": detector.contamination,
    })


def run_model_on_frames(
    model: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int = 42,
    prepare=None,
    dataset: str = "nslkdd",
    protocol: str | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model}")
    fit_on = "all" if model == "rf" else "normal"
    if prepare is None:
        prepare = preprocess
    x_train, y_fit, x_test, y_test, summary = prepare(train, test, fit_on=fit_on)
    extra: dict = {}
    if model == "iforest":
        detector = IsolationForestDetector(seed=seed)
        detector.fit(x_train)
        extra = {"n_estimators": detector.n_estimators, "contamination": detector.contamination}
    elif model == "ocsvm":
        detector = OneClassSVMDetector(seed=seed)
        detector.fit(x_train)
        summary["ocsvm_rows"] = detector.used_rows
        if not summary.get("sampled"):
            summary["sampled"] = detector.used_rows < summary["fit_rows"]
            summary["sample_rows"] = detector.used_rows
        extra = {"nu": detector.nu, "kernel": "rbf", "max_train_rows": detector.max_train_rows}
    elif model == "rf":
        detector = RandomForestDetector(seed=seed)
        detector.fit(x_train, y_fit)
        extra = {"n_estimators": detector.n_estimators, "threshold": 0.5}
    else:
        detector = AutoencoderDetector(seed=seed)
        detector.fit(x_train)
        extra = {
            "epochs": detector.epochs,
            "batch_size": detector.batch_size,
            "learning_rate": detector.learning_rate,
            "final_train_mse": detector.final_loss,
            "threshold": detector.threshold,
            "threshold_quantile": detector.threshold_quantile,
            "architecture": "64-32-16-32-64",
        }
    scores = np.asarray(detector.anomaly_scores(x_test), dtype=float)
    predictions = np.asarray(detector.predict(x_test), dtype=int)
    metrics = evaluate_binary(y_test, predictions, scores)
    record = _record(model, seed, summary, metrics, extra, dataset=dataset, protocol=protocol)
    return record, y_test, scores


def run_nslkdd_model(
    model: str,
    data_dir: Path,
    out_dir: Path,
    seed: int = 42,
    download: bool = True,
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
) -> dict:
    train_df, test_df = _load(data_dir, download, train, test)
    record, y_test, scores = run_model_on_frames(model, train_df, test_df, seed=seed)
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    roc_path = fig_dir / f"nslkdd_{model}_roc.png"
    cm_path = fig_dir / f"nslkdd_{model}_cm.png"
    save_roc(y_test, scores, roc_path, f"NSL-KDD {model} ROC")
    metrics = record["metrics"]
    save_confusion(metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"], cm_path, f"NSL-KDD {model}")
    json_path = out_dir / f"nslkdd_{model}.json"
    md_path = out_dir / f"nslkdd_{model}.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(record), encoding="utf-8")
    record["outputs"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "roc": str(roc_path),
        "confusion": str(cm_path),
    }
    record["_y_test"] = y_test
    record["_scores"] = scores
    return record


def run_nslkdd_all(
    data_dir: Path,
    out_dir: Path,
    seed: int = 42,
    download: bool = True,
) -> list[dict]:
    records = []
    curves = []
    for model in MODEL_NAMES:
        record = run_nslkdd_model(model, data_dir, out_dir, seed=seed, download=download)
        curves.append((model, record.pop("_y_test"), record.pop("_scores")))
        records.append(record)
    overlay = Path(out_dir) / "figures" / "nslkdd_roc_all.png"
    save_roc_overlay(curves, overlay, "NSL-KDD ROC, four detectors")
    table_md = Path(out_dir) / "nslkdd_results.md"
    table_csv = Path(out_dir) / "nslkdd_results.csv"
    table_md.write_text(_table_markdown(records, overlay), encoding="utf-8")
    table_csv.write_text(_table_csv(records, dataset="nslkdd"), encoding="utf-8")
    for record in records:
        record["outputs"]["table_md"] = str(table_md)
        record["outputs"]["table_csv"] = str(table_csv)
        record["outputs"]["roc_overlay"] = str(overlay)
    return records


def _load(data_dir, download, train, test):
    if train is not None and test is not None:
        return train, test
    data_dir = Path(data_dir)
    train_path = data_dir / TRAIN_NAME
    test_path = data_dir / TEST_NAME
    if download and (not train_path.exists() or not test_path.exists()):
        download_nslkdd(data_dir)
    return read_nslkdd_file(train_path), read_nslkdd_file(test_path)


def _record(
    model: str,
    seed: int,
    summary: dict,
    metrics: dict,
    extra: dict,
    dataset: str = "nslkdd",
    protocol: str | None = None,
) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "protocol": protocol or PROTOCOLS[model],
        "counts": summary,
        "settings": extra,
        "metrics": metrics,
    }


def _markdown(record: dict) -> str:
    metrics = record["metrics"]
    counts = record["counts"]
    title = "NSL-KDD" if record["dataset"] == "nslkdd" else "CICIDS2017"
    lines = [
        f"# {title}, {record['model']}",
        "",
        record["protocol"],
        "",
        "| split | rows | normal | attack |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| {counts.get('split_train_label', 'KDDTrain+')} | {counts['train_rows']} | "
            f"{counts['train_normal']} | {counts['train_attack']} |"
        ),
        (
            f"| {counts.get('split_test_label', 'KDDTest+')} | {counts['test_rows']} | "
            f"{counts['test_normal']} | {counts['test_attack']} |"
        ),
        "",
        (
            f"Encoder fit on: {counts['fit_on']} ({counts['fit_rows']} rows). "
            f"Encoded width: {counts['n_features_after_encoding']}. Seed: {record['seed']}."
        ),
        "",
    ]
    if counts.get("sample_note"):
        lines.append(counts["sample_note"])
        lines.append("")
    elif counts.get("sampled"):
        lines.append(
            f"Training sample: {counts.get('sample_rows')} rows. The test file is full."
        )
        lines.append("")
    if counts.get("ocsvm_rows"):
        lines.append(f"One-Class SVM was fit on {counts['ocsvm_rows']} normal training rows.")
        lines.append("")
    lines.extend(
        [
            "| precision | recall | f1 | roc_auc |",
            "| ---: | ---: | ---: | ---: |",
            (
                f"| {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |"
            ),
            "",
            (
                "Confusion matrix (rows true normal/attack, columns predicted): "
                f"TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _table_markdown(records: list[dict], overlay: Path) -> str:
    lines = [
        "# NSL-KDD results",
        "",
        "Attack is the positive class. Test file is full KDDTest+ for every model.",
        "Random Forest sees training labels. The other three do not.",
        "One-Class SVM uses 4000 normal training rows. The others use the full normal training set (Random Forest uses every training row).",
        "",
        "| model | setup | precision | recall | f1 | roc_auc |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    setups = {
        "iforest": "one-class, full normal train",
        "ocsvm": "one-class, 4000 normal rows",
        "rf": "supervised, full train",
        "ae": "one-class, full normal train",
    }
    for record in records:
        metrics = record["metrics"]
        lines.append(
            f"| {record['model']} | {setups[record['model']]} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |"
        )
    lines.extend(["", f"Overlay ROC: `{overlay.as_posix()}`.", ""])
    return "\n".join(lines)


CICIDS_PROTOCOLS = {
    "iforest": (
        "One-class. Isolation Forest is fit only on BENIGN rows in the CICIDS train sample. "
        "Threshold is sklearn's contamination offset (contamination=0.01, 100 trees)."
    ),
    "ocsvm": (
        "One-class. One-Class SVM (RBF, nu=0.01) is fit on at most 4000 BENIGN rows "
        "from the CICIDS train sample. The test sample is not reduced again."
    ),
    "rf": (
        "Supervised reference. Random Forest (100 trees) is fit on the full CICIDS train sample "
        "with the attack label. A score of 0.5 is the decision threshold. "
        "This is not comparable one-to-one with the one-class models."
    ),
    "ae": (
        "One-class. A plain dense autoencoder (64-32-16-32-64, MSE loss, no KL term) "
        "is fit on 90% of the BENIGN rows in the CICIDS train sample for 15 epochs. "
        "The anomaly score is mean squared reconstruction error. "
        "The threshold is the 99th percentile of that error on the held-out 10% of BENIGN rows."
    ),
}


def run_cicids_model(
    model: str,
    data_dir: Path,
    out_dir: Path,
    seed: int = 42,
    download: bool = True,
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    load_summary: dict | None = None,
) -> dict:
    if train is None or test is None:
        train, test, load_summary = load_cicids_splits(data_dir, seed=seed, download=download)
    record, y_test, scores = run_model_on_frames(
        model,
        train,
        test,
        seed=seed,
        prepare=prepare_cicids,
        dataset="cicids2017",
        protocol=CICIDS_PROTOCOLS[model],
    )
    if load_summary:
        record["counts"].update(load_summary)
        record["protocol"] = load_summary["sample_note"] + " " + CICIDS_PROTOCOLS[model]
    return _write_model_outputs(record, y_test, scores, out_dir, "cicids2017", "CICIDS2017")


def run_cicids_all(
    data_dir: Path,
    out_dir: Path,
    seed: int = 42,
    download: bool = True,
) -> list[dict]:
    train, test, load_summary = load_cicids_splits(data_dir, seed=seed, download=download)
    records = []
    curves = []
    for model in MODEL_NAMES:
        record = run_cicids_model(
            model,
            data_dir,
            out_dir,
            seed=seed,
            download=False,
            train=train,
            test=test,
            load_summary=load_summary,
        )
        curves.append((model, record.pop("_y_test"), record.pop("_scores")))
        records.append(record)
    overlay = Path(out_dir) / "figures" / "cicids2017_roc_all.png"
    save_roc_overlay(curves, overlay, "CICIDS2017 ROC, four detectors")
    table_md = Path(out_dir) / "cicids2017_results.md"
    table_csv = Path(out_dir) / "cicids2017_results.csv"
    note = load_summary["sample_note"]
    table_md.write_text(_dataset_table_markdown("CICIDS2017 results", records, overlay, note), encoding="utf-8")
    table_csv.write_text(_table_csv(records, dataset="cicids2017"), encoding="utf-8")
    for record in records:
        record["outputs"]["table_md"] = str(table_md)
        record["outputs"]["table_csv"] = str(table_csv)
        record["outputs"]["roc_overlay"] = str(overlay)
    return records


def run_all_datasets(data_dir: Path, out_dir: Path, seed: int = 42, download: bool = True) -> list[dict]:
    nsl_records = run_nslkdd_all(data_dir, out_dir, seed=seed, download=download)
    cic_records = run_cicids_all(data_dir, out_dir, seed=seed, download=download)
    combined = nsl_records + cic_records
    out_dir = Path(out_dir)
    table_md = out_dir / "run_all.md"
    table_csv = out_dir / "run_all.csv"
    table_md.write_text(_combined_markdown(combined), encoding="utf-8")
    table_csv.write_text(_table_csv(combined, dataset=None), encoding="utf-8")
    for record in combined:
        record["outputs"]["run_all_md"] = str(table_md)
        record["outputs"]["run_all_csv"] = str(table_csv)
    return combined


def _write_model_outputs(record, y_test, scores, out_dir, slug: str, title: str) -> dict:
    out_dir = Path(out_dir)
    model = record["model"]
    fig_dir = out_dir / "figures"
    roc_path = fig_dir / f"{slug}_{model}_roc.png"
    cm_path = fig_dir / f"{slug}_{model}_cm.png"
    save_roc(y_test, scores, roc_path, f"{title} {model} ROC")
    metrics = record["metrics"]
    save_confusion(metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"], cm_path, f"{title} {model}")
    json_path = out_dir / f"{slug}_{model}.json"
    md_path = out_dir / f"{slug}_{model}.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(record), encoding="utf-8")
    record["outputs"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "roc": str(roc_path),
        "confusion": str(cm_path),
    }
    record["_y_test"] = y_test
    record["_scores"] = scores
    return record


def _dataset_table_markdown(title: str, records: list[dict], overlay: Path, note: str) -> str:
    lines = [
        f"# {title}",
        "",
        note,
        "",
        "Random Forest sees training labels. The other three do not.",
        "One-Class SVM uses at most 4000 normal training rows.",
        "",
        "| model | precision | recall | f1 | roc_auc |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        metrics = record["metrics"]
        lines.append(
            f"| {record['model']} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |"
        )
    lines.extend(["", f"Overlay ROC: `{overlay.as_posix()}`.", ""])
    return "\n".join(lines)


def _combined_markdown(records: list[dict]) -> str:
    lines = [
        "# Run-all results",
        "",
        "Attack is the positive class. NSL-KDD uses the official KDDTest+ file with no row sampling.",
        "CICIDS2017 uses a reservoir sample and an 80/20 split defined in this repo. See cicids2017_results.md.",
        "Random Forest is supervised. The other models are one-class. Do not read a higher Random Forest score as a like-for-like win.",
        "",
        "| dataset | model | precision | recall | f1 | roc_auc |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        metrics = record["metrics"]
        lines.append(
            f"| {record['dataset']} | {record['model']} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _table_csv(records: list[dict], dataset: str | None = None) -> str:
    include_dataset = dataset is None
    header = "dataset,model,precision,recall,f1,roc_auc,tn,fp,fn,tp" if include_dataset else "model,precision,recall,f1,roc_auc,tn,fp,fn,tp"
    lines = [header]
    for record in records:
        metrics = record["metrics"]
        prefix = f"{record['dataset']}," if include_dataset else ""
        if dataset is not None:
            prefix = ""
        lines.append(
            prefix
            + "{model},{precision:.6f},{recall:.6f},{f1:.6f},{roc_auc:.6f},{tn},{fp},{fn},{tp}".format(
                model=record["model"], **metrics
            )
        )
    return "\n".join(lines) + "\n"


def run_nslkdd_iforest(data_dir: Path, out_dir: Path, seed: int = 42, download: bool = True) -> dict:
    record = run_nslkdd_model("iforest", data_dir, out_dir, seed=seed, download=download)
    record.pop("_y_test", None)
    record.pop("_scores", None)
    return record
