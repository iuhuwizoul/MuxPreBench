import numpy as np

from multiplexbench.data import generate_sample
from multiplexbench.metrics import technical_metrics
from multiplexbench.preprocessing import PreprocessingPipeline, subtract_background


def test_morphological_background_subtraction_removes_smooth_offset() -> None:
    image = np.full((48, 48), 0.2, dtype=np.float32)
    image[20:28, 20:28] += 0.8

    corrected = subtract_background(image, method="morphology", radius=6)

    assert np.mean(corrected[:10, :10]) < 1e-5
    assert np.mean(corrected[21:27, 21:27]) > 0.5


def test_phase_registration_reduces_known_shift_error() -> None:
    sample = generate_sample(1, image_size=72, n_cells=12, severity=1.0, seed=17)
    raw = PreprocessingPipeline().transform(sample)
    registered = PreprocessingPipeline(registration="phase").transform(sample)

    raw_error = technical_metrics(sample, raw)["registration_error_px"]
    registered_error = technical_metrics(sample, registered)["registration_error_px"]

    assert registered_error < raw_error


def test_pipeline_outputs_finite_values() -> None:
    sample = generate_sample(0, image_size=64, n_cells=9, severity=1.0, seed=3)
    output = PreprocessingPipeline(
        background="morphology", illumination="gaussian", registration="phase"
    ).transform(sample)

    assert output.markers.shape == sample.clean_markers.shape
    assert np.isfinite(output.markers).all()
    assert np.isfinite(list(technical_metrics(sample, output).values())).all()
