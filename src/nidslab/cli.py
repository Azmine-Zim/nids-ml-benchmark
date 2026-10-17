"""Command line for NIDS-Lab."""

from __future__ import annotations

from pathlib import Path

import typer

from nidslab.data.nslkdd import download_nslkdd
from nidslab.pipeline import MODEL_NAMES, run_nslkdd_all, run_nslkdd_model

app = typer.Typer(help="Classical ML benchmark for network intrusion detection.", no_args_is_help=True)


def _echo_metrics(record: dict) -> None:
    metrics = record["metrics"]
    typer.echo(
        "{model}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} roc_auc={roc_auc:.4f}".format(
            model=record["model"], **metrics
        )
    )
    typer.echo("TN={tn} FP={fp} FN={fn} TP={tp}".format(**metrics))


@app.command()
def download(
    dataset: str = typer.Option("nslkdd", "--dataset", help="Dataset name."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Where to store raw files."),
) -> None:
    """Download a public dataset into the data folder."""
    if dataset != "nslkdd":
        raise typer.BadParameter("Only nslkdd can be downloaded so far. cicids2017 comes later.")
    train_path, test_path = download_nslkdd(data_dir)
    typer.echo(f"train: {train_path}")
    typer.echo(f"test:  {test_path}")


@app.command()
def run(
    dataset: str = typer.Option("nslkdd", "--dataset", help="Dataset name."),
    model: str = typer.Option("iforest", "--model", help="iforest, ocsvm, rf, or ae."),
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Train one detector and write metrics and figures."""
    if dataset != "nslkdd":
        raise typer.BadParameter("Only nslkdd is wired up so far.")
    if model not in MODEL_NAMES:
        raise typer.BadParameter("model must be one of: " + ", ".join(MODEL_NAMES))
    record = run_nslkdd_model(model, data_dir=data_dir, out_dir=out_dir, seed=seed, download=True)
    record.pop("_y_test", None)
    record.pop("_scores", None)
    _echo_metrics(record)
    typer.echo(f"wrote {record['outputs']['json']}")
    typer.echo(f"wrote {record['outputs']['roc']}")


@app.command("run-all")
def run_all(
    dataset: str = typer.Option("nslkdd", "--dataset"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Train every detector and write a results table."""
    if dataset != "nslkdd":
        raise typer.BadParameter("run-all currently covers nslkdd only.")
    records = run_nslkdd_all(data_dir=data_dir, out_dir=out_dir, seed=seed, download=True)
    for record in records:
        _echo_metrics(record)
    typer.echo(f"wrote {records[0]['outputs']['table_md']}")
    typer.echo(f"wrote {records[0]['outputs']['table_csv']}")
