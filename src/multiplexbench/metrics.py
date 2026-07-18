"""Technical image-quality metrics and uncertainty summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from .data import SyntheticSample
from .preprocessing import PipelineOutput

TECHNICAL_METRICS = (
    "ssim",
    "signal_pearson",
    "signal_nrmse",
    "background_fraction",
    "registration_error_px",
    "runtime_ms",
)


def robust_normalize(image: np.ndarray) -> np.ndarray:
    """Map an image to [0, 1] without letting rare hot pixels set the range."""

    low, high = np.quantile(image, [0.01, 0.995])
    if high <= low + 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def technical_metrics(sample: SyntheticSample, output: PipelineOutput) -> dict[str, float]:
    """Compare a processed stack with its known clean and acquisition ground truth."""

    foreground = sample.foreground_mask
    background = ~foreground
    ssim_values: list[float] = []
    correlations: list[float] = []
    nrmse_values: list[float] = []
    background_values: list[float] = []

    for clean_image, processed_image in zip(
        sample.clean_markers, output.markers, strict=True
    ):
        clean = robust_normalize(clean_image)
        processed = robust_normalize(processed_image)
        ssim_values.append(
            float(structural_similarity(clean, processed, data_range=1.0))
        )
        correlations.append(_safe_pearson(clean[foreground], processed[foreground]))
        squared_error = (clean[foreground] - processed[foreground]) ** 2
        nrmse_values.append(float(np.sqrt(np.mean(squared_error))))
        signal_mean = float(np.mean(processed[foreground]))
        background_values.append(
            float(np.mean(processed[background])) / (signal_mean + 1e-8)
        )

    expected_corrections = -sample.acquisition_shifts
    registration_error = np.linalg.norm(
        output.correction_shifts[1:] - expected_corrections[1:], axis=1
    )
    return {
        "ssim": float(np.mean(ssim_values)),
        "signal_pearson": float(np.mean(correlations)),
        "signal_nrmse": float(np.mean(nrmse_values)),
        "background_fraction": float(np.mean(background_values)),
        "registration_error_px": float(np.mean(registration_error)),
        "runtime_ms": output.runtime_ms,
    }


def bootstrap_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: tuple[str, ...] | list[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Return a tidy non-parametric 95% confidence interval per group."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    grouped = frame.groupby(group_columns, sort=False, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys, strict=True))
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            draws = rng.choice(values, size=(iterations, values.size), replace=True)
            means = np.mean(draws, axis=1)
            rows.append(
                {
                    **key_values,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "ci95_low": float(np.quantile(means, 0.025)),
                    "ci95_high": float(np.quantile(means, 0.975)),
                    "n": int(values.size),
                }
            )
    return pd.DataFrame(rows)
