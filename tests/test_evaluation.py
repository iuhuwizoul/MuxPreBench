import numpy as np

from multiplexbench.data import generate_dataset
from multiplexbench.evaluation import DOWNSTREAM_METRICS, evaluate_downstream


def test_downstream_evaluation_is_group_aware_and_complete() -> None:
    samples = generate_dataset(
        n_samples=6, image_size=48, n_cells=9, severity=0.8, seed=29
    )
    stacks = {"clean": [sample.clean_markers for sample in samples]}

    result = evaluate_downstream(
        samples,
        stacks,
        held_out_channel=3,
        repeats=1,
        test_fraction=0.34,
        pixels_per_image=80,
        seed=29,
    )

    assert result.shape[0] == 1
    assert set(DOWNSTREAM_METRICS).issubset(result.columns)
    assert np.isfinite(result[list(DOWNSTREAM_METRICS)].to_numpy()).all()
