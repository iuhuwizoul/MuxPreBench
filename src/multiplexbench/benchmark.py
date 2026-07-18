"""End-to-end benchmark orchestration and command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import CHANNEL_NAMES, generate_dataset
from .evaluation import DOWNSTREAM_METRICS, evaluate_downstream
from .metrics import TECHNICAL_METRICS, bootstrap_summary, technical_metrics
from .preprocessing import PreprocessingPipeline
from .visualization import (
    plot_downstream,
    plot_example,
    plot_robustness,
    plot_technical_benchmark,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The benchmark configuration must be a mapping.")
    return config


def _make_pipelines(config: dict[str, Any]) -> dict[str, PreprocessingPipeline]:
    common = config["preprocessing"]
    return {
        name: PreprocessingPipeline(**methods, **common)
        for name, methods in config["pipelines"].items()
    }


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-image", "scikit-learn", "matplotlib")
    return {package: metadata.version(package) for package in packages}


def _paired_improvements(
    frame: pd.DataFrame,
    unit: str,
    metrics: dict[str, str],
    candidate: str,
    iterations: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Bootstrap candidate-vs-raw improvements while preserving pairing."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for metric, direction in metrics.items():
        pivot = frame.pivot(index=unit, columns="pipeline", values=metric).dropna()
        difference = pivot[candidate] - pivot["raw"]
        if direction == "lower":
            difference = -difference
        values = difference.to_numpy(dtype=float)
        draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
        rows.append(
            {
                "candidate": candidate,
                "metric": metric,
                "direction": direction,
                "mean_improvement": float(np.mean(values)),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "n_pairs": int(len(values)),
            }
        )
    return rows


def run_benchmark(config: dict[str, Any], smoke: bool = False) -> Path:
    config = json.loads(json.dumps(config))  # defensive deep copy of YAML-native values
    if smoke:
        config["data"]["n_samples"] = 6
        config["data"]["severities"] = [1.0]
        config["evaluation"]["bootstrap_iterations"] = 100
        config["evaluation"]["downstream_repeats"] = 1
        config["evaluation"]["pixels_per_image"] = 150
        config["output_dir"] = str(Path(config["output_dir"]) / "tmp" / "smoke")

    seed = int(config["seed"])
    data_config = config["data"]
    evaluation = config["evaluation"]
    output_dir = Path(config["output_dir"])
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pipelines = _make_pipelines(config)

    technical_rows: list[dict[str, float | int | str]] = []
    reference_samples = None
    reference_stacks: dict[str, list[np.ndarray]] = {}
    example_stacks: dict[str, np.ndarray] = {}

    for severity in data_config["severities"]:
        samples = generate_dataset(
            n_samples=int(data_config["n_samples"]),
            image_size=int(data_config["image_size"]),
            n_cells=int(data_config["cells_per_image"]),
            severity=float(severity),
            seed=seed,
        )
        is_reference = np.isclose(float(severity), float(evaluation["reference_severity"]))
        if is_reference:
            reference_samples = samples
            reference_stacks = {name: [] for name in pipelines}

        for sample in samples:
            for pipeline_name, pipeline in pipelines.items():
                output = pipeline.transform(sample)
                technical_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "severity": float(severity),
                        "pipeline": pipeline_name,
                        **technical_metrics(sample, output),
                    }
                )
                if is_reference:
                    reference_stacks[pipeline_name].append(output.markers)
                    if sample.sample_id == 0 and pipeline_name in {"raw", "full"}:
                        example_stacks[pipeline_name] = output.markers

    if reference_samples is None:
        raise ValueError("reference_severity must be included in data.severities")

    technical = pd.DataFrame(technical_rows)
    technical.to_csv(output_dir / "technical_metrics.csv", index=False)
    technical_summary = bootstrap_summary(
        technical,
        ["severity", "pipeline"],
        TECHNICAL_METRICS,
        int(evaluation["bootstrap_iterations"]),
        seed,
    )
    technical_summary.to_csv(output_dir / "technical_summary.csv", index=False)

    held_out = CHANNEL_NAMES.index(str(evaluation["held_out_channel"]))
    downstream = evaluate_downstream(
        reference_samples,
        reference_stacks,
        held_out_channel=held_out,
        repeats=int(evaluation["downstream_repeats"]),
        test_fraction=float(evaluation["test_fraction"]),
        pixels_per_image=int(evaluation["pixels_per_image"]),
        seed=seed,
    )
    downstream.to_csv(output_dir / "downstream_metrics.csv", index=False)
    downstream_summary = bootstrap_summary(
        downstream,
        ["pipeline"],
        DOWNSTREAM_METRICS,
        int(evaluation["bootstrap_iterations"]),
        seed + 1,
    )
    downstream_summary.to_csv(output_dir / "downstream_summary.csv", index=False)

    candidate = "full_no_illumination"
    reference_technical = technical[
        np.isclose(technical["severity"], float(evaluation["reference_severity"]))
    ]
    improvement_rows = _paired_improvements(
        reference_technical,
        "sample_id",
        {
            "ssim": "higher",
            "signal_pearson": "higher",
            "signal_nrmse": "lower",
            "background_fraction": "lower",
            "registration_error_px": "lower",
        },
        candidate,
        int(evaluation["bootstrap_iterations"]),
        seed + 2,
    )
    improvement_rows.extend(
        _paired_improvements(
            downstream,
            "repeat",
            {
                "protein_pearson": "higher",
                "cell_type_balanced_accuracy": "higher",
                "cell_stage_r2": "higher",
                "cell_stage_mae": "lower",
            },
            candidate,
            int(evaluation["bootstrap_iterations"]),
            seed + 3,
        )
    )
    pd.DataFrame(improvement_rows).to_csv(output_dir / "paired_improvements.csv", index=False)

    sample = reference_samples[0]
    np.savez_compressed(
        output_dir / "example_field.npz",
        clean=sample.clean_markers,
        corrupted=example_stacks["raw"],
        processed=example_stacks["full"],
        cell_labels=sample.cell_labels,
        acquisition_shifts=sample.acquisition_shifts,
        channels=np.asarray(CHANNEL_NAMES),
    )
    plot_example(
        sample,
        example_stacks["raw"],
        example_stacks["full"],
        figures_dir / "example_field.png",
    )
    plot_technical_benchmark(
        technical_summary,
        float(evaluation["reference_severity"]),
        figures_dir / "technical_benchmark.png",
    )
    plot_robustness(technical_summary, figures_dir / "robustness.png")
    plot_downstream(downstream_summary, figures_dir / "downstream.png")

    canonical_config = json.dumps(config, sort_keys=True).encode()
    metadata_payload = {
        "config_sha256": hashlib.sha256(canonical_config).hexdigest(),
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "synthetic_data": True,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a six-sample end-to-end check in results/tmp/smoke.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_benchmark(load_config(args.config), smoke=args.smoke)
    print(f"Benchmark complete: {output}")


if __name__ == "__main__":
    main()
