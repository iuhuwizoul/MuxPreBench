import numpy as np

from multiplexbench.data import generate_sample


def test_generation_is_reproducible_and_paired_across_severity() -> None:
    first = generate_sample(2, image_size=64, n_cells=9, severity=0.5, seed=11)
    repeated = generate_sample(2, image_size=64, n_cells=9, severity=0.5, seed=11)
    stronger = generate_sample(2, image_size=64, n_cells=9, severity=1.5, seed=11)

    np.testing.assert_array_equal(first.observed_cycles, repeated.observed_cycles)
    np.testing.assert_array_equal(first.clean_markers, stronger.clean_markers)
    assert first.observed_cycles.shape == (4, 2, 64, 64)
    assert first.clean_markers.shape == (4, 64, 64)


def test_labels_and_biological_targets_are_well_formed() -> None:
    sample = generate_sample(0, image_size=64, n_cells=9, seed=7)

    assert sample.cell_labels.max() == 9
    assert set(np.unique(sample.cell_types[1:])) == {0, 1}
    assert np.all((sample.cell_stages[1:] > 0) & (sample.cell_stages[1:] < 1))
    assert np.count_nonzero(sample.foreground_mask) > 0
