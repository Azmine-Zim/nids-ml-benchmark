"""ROC curves and confusion matrices."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def save_roc(y_true: np.ndarray, scores: np.ndarray, path: Path, title: str) -> None:
    fpr, tpr, _thresholds = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1f4e79", lw=2)
    ax.plot([0, 1], [0, 1], color="#888888", lw=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion(tn: int, fp: int, fn: int, tp: int, path: Path, title: str) -> None:
    matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks([0, 1], ["pred normal", "pred attack"])
    ax.set_yticks([0, 1], ["true normal", "true attack"])
    ax.set_title(title)
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color="black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc_overlay(curves: list[tuple[str, np.ndarray, np.ndarray]], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, y_true, scores in curves:
        fpr, tpr, _thresholds = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, lw=2, label=name)
    ax.plot([0, 1], [0, 1], color="#888888", lw=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
