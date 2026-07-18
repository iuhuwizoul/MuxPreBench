"""Group-aware downstream tests for biological signal preservation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import SyntheticSample
from .metrics import robust_normalize

DOWNSTREAM_METRICS = (
    "protein_r2",
    "protein_pearson",
    "cell_type_balanced_accuracy",
    "cell_type_f1_macro",
    "cell_stage_r2",
    "cell_stage_mae",
)


def _normalize_stack(stack: np.ndarray) -> np.ndarray:
    return np.stack([robust_normalize(channel) for channel in stack])


def _cell_table(
    sample: SyntheticSample, stack: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = _normalize_stack(stack)
    features: list[list[float]] = []
    cell_types: list[int] = []
    stages: list[float] = []
    for cell_id in range(1, len(sample.cell_types)):
        mask = sample.cell_labels == cell_id
        if not np.any(mask):
            continue
        row: list[float] = []
        for channel in normalized:
            values = channel[mask]
            row.extend((float(np.mean(values)), float(np.std(values))))
        features.append(row)
        cell_types.append(int(sample.cell_types[cell_id]))
        stages.append(float(sample.cell_stages[cell_id]))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(cell_types),
        np.asarray(stages, dtype=np.float32),
    )


def _pixel_features(stack: np.ndarray, held_out: int) -> np.ndarray:
    normalized = _normalize_stack(stack)
    available = np.delete(normalized, held_out, axis=0)
    local_context = np.stack([ndi.gaussian_filter(channel, sigma=1.5) for channel in available])
    return np.concatenate([available, local_context], axis=0).reshape(2 * len(available), -1).T


def _sample_indices(
    rng: np.random.Generator, mask: np.ndarray, n_pixels: int
) -> np.ndarray:
    foreground = np.flatnonzero(mask.ravel())
    background = np.flatnonzero(~mask.ravel())
    n_foreground = min(len(foreground), n_pixels // 2)
    n_background = min(len(background), n_pixels - n_foreground)
    selected_foreground = rng.choice(foreground, size=n_foreground, replace=False)
    selected_background = rng.choice(background, size=n_background, replace=False)
    return np.concatenate([selected_foreground, selected_background])


def _concatenate_cell_data(
    samples: list[SyntheticSample],
    stacks: list[np.ndarray],
    indices: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tables = [_cell_table(samples[index], stacks[index]) for index in indices]
    return tuple(  # type: ignore[return-value]
        np.concatenate(columns, axis=0) for columns in zip(*tables, strict=True)
    )


def _evaluate_cells(
    samples: list[SyntheticSample],
    stacks: list[np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> dict[str, float]:
    x_train, type_train, stage_train = _concatenate_cell_data(samples, stacks, train_indices)
    x_test, type_test, stage_test = _concatenate_cell_data(samples, stacks, test_indices)

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    classifier.fit(x_train, type_train)
    type_prediction = classifier.predict(x_test)

    regressor = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    regressor.fit(x_train, stage_train)
    stage_prediction = regressor.predict(x_test)
    return {
        "cell_type_balanced_accuracy": float(
            balanced_accuracy_score(type_test, type_prediction)
        ),
        "cell_type_f1_macro": float(f1_score(type_test, type_prediction, average="macro")),
        "cell_stage_r2": float(r2_score(stage_test, stage_prediction)),
        "cell_stage_mae": float(mean_absolute_error(stage_test, stage_prediction)),
    }


def _evaluate_held_out_protein(
    samples: list[SyntheticSample],
    stacks: list[np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    held_out: int,
    pixels_per_image: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    train_features: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    for index in train_indices:
        features = _pixel_features(stacks[index], held_out)
        target = robust_normalize(samples[index].clean_markers[held_out]).ravel()
        selected = _sample_indices(rng, samples[index].foreground_mask, pixels_per_image)
        train_features.append(features[selected])
        train_targets.append(target[selected])

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=60,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=seed,
    )
    model.fit(np.concatenate(train_features), np.concatenate(train_targets))

    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for index in test_indices:
        mask = samples[index].foreground_mask.ravel()
        target = robust_normalize(samples[index].clean_markers[held_out]).ravel()[mask]
        prediction = model.predict(_pixel_features(stacks[index], held_out))[mask]
        targets.append(target)
        predictions.append(prediction)

    target_values = np.concatenate(targets)
    prediction_values = np.concatenate(predictions)
    correlation = float(np.corrcoef(target_values, prediction_values)[0, 1])
    return {
        "protein_r2": float(r2_score(target_values, prediction_values)),
        "protein_pearson": correlation,
    }


def evaluate_downstream(
    samples: list[SyntheticSample],
    processed_stacks: dict[str, list[np.ndarray]],
    held_out_channel: int,
    repeats: int,
    test_fraction: float,
    pixels_per_image: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate all pipelines with identical group-level train/test splits."""

    groups = np.arange(len(samples))
    rows: list[dict[str, float | int | str]] = []
    for repeat in range(repeats):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=seed + repeat,
        )
        train_indices, test_indices = next(splitter.split(groups, groups=groups))
        for pipeline_name, stacks in processed_stacks.items():
            cell_metrics = _evaluate_cells(
                samples, stacks, train_indices, test_indices, seed + repeat
            )
            protein_metrics = _evaluate_held_out_protein(
                samples,
                stacks,
                train_indices,
                test_indices,
                held_out_channel,
                pixels_per_image,
                seed + repeat,
            )
            rows.append(
                {
                    "pipeline": pipeline_name,
                    "repeat": repeat,
                    **cell_metrics,
                    **protein_metrics,
                }
            )
    return pd.DataFrame(rows)
