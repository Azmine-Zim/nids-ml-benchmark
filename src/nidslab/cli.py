"""Command line for NIDS-Lab."""

from __future__ import annotations

from pathlib import Path

import typer

from nidslab.data.cicids import download_cicids
from nidslab.data.nslkdd import download_nslkdd
from nidslab.pipeline import (
    MODEL_NAMES,
    run_all_datasets,
    run_cicids_all,
    run_cicids_model,
    run_nslkdd_all,
    run_nslkdd_model,
)

app = typer.Typer(help="Classical ML benchmark for network intrusion detection.", no_args_is_help=True)
DATASETS = ("nslkdd", "cicids2017", "all")


def _echo_metrics(record: dict) -> None:
    metrics = record["metrics"]
    typer.echo(
        "{dataset} {model}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} roc_auc={roc_auc:.4f}".format(
            dataset=record["dataset"], model=record["model"], **metrics
        )
    )
    typer.echo("TN={tn} FP={fp} FN={fn} TP={tp}".format(**metrics))


@app.command()
def download(
    dataset: str = typer.Option("nslkdd", "--dataset", help="nslkdd or cicids2017."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Where to store raw files."),
) -> None:
    """Download a public dataset into the data folder."""
    if dataset == "nslkdd":
        train_path, test_path = download_nslkdd(data_dir)
        typer.echo(f"train: {train_path}")
        typer.echo(f"test:  {test_path}")
        return
    if dataset == "cicids2017":
        folder = download_cicids(data_dir)
        typer.echo(f"cicids: {folder}")
        return
    raise typer.BadParameter("dataset must be nslkdd or cicids2017.")


@app.command()
def run(
    dataset: str = typer.Option("nslkdd", "--dataset", help="nslkdd or cicids2017."),
    model: str = typer.Option("iforest", "--model", help="iforest, ocsvm, rf, or ae."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Train one detector and write metrics and figures."""
    if model not in MODEL_NAMES:
        raise typer.BadParameter("model must be one of: " + ", ".join(MODEL_NAMES))
    if dataset == "nslkdd":
        record = run_nslkdd_model(model, data_dir=data_dir, out_dir=out_dir, seed=seed, download=True)
    elif dataset == "cicids2017":
        record = run_cicids_model(model, data_dir=data_dir, out_dir=out_dir, seed=seed, download=True)
    else:
        raise typer.BadParameter("dataset must be nslkdd or cicids2017.")
    record.pop("_y_test", None)
    record.pop("_scores", None)
    _echo_metrics(record)
    typer.echo(f"wrote {record['outputs']['json']}")
    typer.echo(f"wrote {record['outputs']['roc']}")


@app.command("run-all")
def run_all(
    dataset: str = typer.Option("all", "--dataset", help="nslkdd, cicids2017, or all."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Train every detector and write a results table."""
    if dataset == "nslkdd":
        records = run_nslkdd_all(data_dir=data_dir, out_dir=out_dir, seed=seed, download=True)
        table_key = "table_md"
    elif dataset == "cicids2017":
        records = run_cicids_all(data_dir=data_dir, out_dir=out_dir, seed=seed, download=True)
        table_key = "table_md"
    elif dataset == "all":
        records = run_all_datasets(data_dir=data_dir, out_dir=out_dir, seed=seed, download=True)
        table_key = "run_all_md"
    else:
        raise typer.BadParameter("dataset must be one of: " + ", ".join(DATASETS))
    for record in records:
        _echo_metrics(record)
    typer.echo(f"wrote {records[0]['outputs'][table_key]}")
