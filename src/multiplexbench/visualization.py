"""Research-report figures generated directly from benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import CHANNEL_NAMES, SyntheticSample  # noqa: E402
from .metrics import robust_normalize  # noqa: E402

COLORS = {
    "raw": "#9CA3AF",
    "background_only": "#60A5FA",
    "illumination_only": "#A78BFA",
    "registration_only": "#F59E0B",
    "full": "#059669",
    "full_no_background": "#38BDF8",
    "full_no_illumination": "#C084FC",
    "full_no_registration": "#F97316",
}

DISPLAY_NAMES = {
    "raw": "raw",
    "background_only": "background",
    "illumination_only": "illumination",
    "registration_only": "registration",
    "full": "all three",
    "full_no_background": "all − background",
    "full_no_illumination": "background + registration",
    "full_no_registration": "background + illumination",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_example(
    sample: SyntheticSample,
    raw_stack: np.ndarray,
    full_stack: np.ndarray,
    path: Path,
) -> None:
    rows = (
        ("Clean ground truth", sample.clean_markers),
        ("Corrupted", raw_stack),
        ("Full", full_stack),
    )
    fig, axes = plt.subplots(3, len(CHANNEL_NAMES), figsize=(11, 7.2))
    for row, (row_name, stack) in enumerate(rows):
        for column, channel_name in enumerate(CHANNEL_NAMES):
            axes[row, column].imshow(robust_normalize(stack[column]), cmap="magma", vmin=0, vmax=1)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(channel_name, fontsize=10, fontweight="bold")
            if column == 0:
                axes[row, column].set_ylabel(row_name, fontsize=10, fontweight="bold")
    fig.suptitle("Synthetic iterative immunofluorescence: one field of view", fontsize=14)
    fig.tight_layout()
    _save(fig, path)


def _summary_value(
    summary: pd.DataFrame, pipeline: str, metric: str, severity: float | None = None
) -> tuple[float, float, float]:
    selected = summary[(summary["pipeline"] == pipeline) & (summary["metric"] == metric)]
    if severity is not None and "severity" in selected.columns:
        selected = selected[np.isclose(selected["severity"].astype(float), severity)]
    row = selected.iloc[0]
    return float(row["mean"]), float(row["ci95_low"]), float(row["ci95_high"])


def plot_technical_benchmark(
    summary: pd.DataFrame, reference_severity: float, path: Path
) -> None:
    pipelines = [
        "raw",
        "background_only",
        "illumination_only",
        "registration_only",
        "full",
        "full_no_illumination",
    ]
    metrics = [
        ("ssim", "SSIM ↑"),
        ("signal_pearson", "Signal correlation ↑"),
        ("background_fraction", "Background / signal ↓"),
        ("registration_error_px", "Registration error (px) ↓"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for axis, (metric, label) in zip(axes.ravel(), metrics, strict=True):
        values, lower, upper = [], [], []
        for pipeline in pipelines:
            mean, low, high = _summary_value(summary, pipeline, metric, reference_severity)
            values.append(mean)
            lower.append(mean - low)
            upper.append(high - mean)
        x = np.arange(len(pipelines))
        axis.bar(
            x,
            values,
            color=[COLORS[pipeline] for pipeline in pipelines],
            yerr=np.asarray([lower, upper]),
            capsize=3,
        )
        axis.set_xticks(
            x,
            [DISPLAY_NAMES[name].replace(" + ", "\n+ ") for name in pipelines],
            fontsize=7,
        )
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Technical benchmark at artifact severity {reference_severity:g}", fontsize=14)
    fig.tight_layout()
    _save(fig, path)


def plot_robustness(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, metric, label in (
        (axes[0], "ssim", "SSIM ↑"),
        (axes[1], "registration_error_px", "Registration error (px) ↓"),
    ):
        for pipeline in ("raw", "full", "full_no_illumination"):
            selected = summary[
                (summary["pipeline"] == pipeline) & (summary["metric"] == metric)
            ].sort_values("severity")
            axis.plot(
                selected["severity"],
                selected["mean"],
                marker="o",
                linewidth=2,
                color=COLORS[pipeline],
                label=DISPLAY_NAMES[pipeline],
            )
            axis.fill_between(
                selected["severity"].astype(float),
                selected["ci95_low"].astype(float),
                selected["ci95_high"].astype(float),
                color=COLORS[pipeline],
                alpha=0.14,
            )
        axis.set_xlabel("Artifact severity")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Robustness under increasing acquisition artifacts", fontsize=14)
    fig.tight_layout()
    _save(fig, path)


def plot_downstream(summary: pd.DataFrame, path: Path) -> None:
    pipelines = [
        "raw",
        "background_only",
        "illumination_only",
        "registration_only",
        "full",
        "full_no_illumination",
    ]
    metrics = [
        ("protein_pearson", "Held-out ERK\ncorrelation ↑"),
        ("cell_type_balanced_accuracy", "Cell type\nbalanced accuracy ↑"),
        ("cell_stage_r2", "Cell stage\n$R^2$ ↑"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        values, lower, upper = [], [], []
        for pipeline in pipelines:
            mean, low, high = _summary_value(summary, pipeline, metric)
            values.append(mean)
            lower.append(mean - low)
            upper.append(high - mean)
        x = np.arange(len(pipelines))
        axis.bar(
            x,
            values,
            color=[COLORS[pipeline] for pipeline in pipelines],
            yerr=np.asarray([lower, upper]),
            capsize=3,
        )
        axis.set_xticks(
            x,
            [DISPLAY_NAMES[name].replace(" + ", "\n+ ") for name in pipelines],
            fontsize=7,
        )
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Downstream biological signal-preservation tests", fontsize=14)
    fig.tight_layout()
    _save(fig, path)
