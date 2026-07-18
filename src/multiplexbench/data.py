"""Synthetic multiplexed imaging data with known technical ground truth.

Each acquisition cycle contains a repeated DNA anchor and one protein marker.
The same spatial shift is applied to both images in a cycle, mirroring a common
registration strategy in iterative immunofluorescence experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

CHANNEL_NAMES = ("DNA", "MEMBRANE", "ACTIN", "ERK")


@dataclass(frozen=True)
class SyntheticSample:
    """One field of view and its corruption ground truth."""

    sample_id: int
    clean_markers: np.ndarray  # (cycles, height, width), spatially aligned
    observed_cycles: np.ndarray  # (cycles, [anchor, marker], height, width)
    cell_labels: np.ndarray
    nucleus_labels: np.ndarray
    cell_types: np.ndarray  # indexed by integer cell id
    cell_stages: np.ndarray  # continuous target, indexed by integer cell id
    acquisition_shifts: np.ndarray  # (cycles, [dy, dx])
    severity: float

    @property
    def foreground_mask(self) -> np.ndarray:
        return self.cell_labels > 0


def _draw_cells(
    rng: np.random.Generator, image_size: int, n_cells: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:image_size, :image_size]
    cell_labels = np.zeros((image_size, image_size), dtype=np.int16)
    nucleus_labels = np.zeros_like(cell_labels)
    cell_types = np.zeros(n_cells + 1, dtype=np.int8)
    cell_stages = np.zeros(n_cells + 1, dtype=np.float32)

    grid_width = int(np.ceil(np.sqrt(n_cells)))
    coordinates = np.linspace(10, image_size - 11, grid_width)
    centers = [(y, x) for y in coordinates for x in coordinates][:n_cells]

    for cell_id, (base_y, base_x) in enumerate(centers, start=1):
        cy = float(base_y + rng.uniform(-2.2, 2.2))
        cx = float(base_x + rng.uniform(-2.2, 2.2))
        radius = float(rng.uniform(5.5, 7.2))
        nucleus_radius = radius * float(rng.uniform(0.38, 0.52))
        distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        available = cell_labels == 0
        cell_mask = (distance <= radius) & available
        nucleus_mask = (distance <= nucleus_radius) & cell_mask
        cell_labels[cell_mask] = cell_id
        nucleus_labels[nucleus_mask] = cell_id
        cell_types[cell_id] = rng.integers(0, 2)
        cell_stages[cell_id] = rng.uniform(0.05, 0.95)

    return cell_labels, nucleus_labels, cell_types, cell_stages


def _render_clean_markers(
    rng: np.random.Generator,
    cell_labels: np.ndarray,
    nucleus_labels: np.ndarray,
    cell_types: np.ndarray,
    cell_stages: np.ndarray,
) -> np.ndarray:
    shape = cell_labels.shape
    markers = np.zeros((len(CHANNEL_NAMES), *shape), dtype=np.float32)

    for cell_id in range(1, len(cell_types)):
        cell = cell_labels == cell_id
        nucleus = nucleus_labels == cell_id
        if not np.any(cell):
            continue
        cytoplasm = cell & ~nucleus
        inner = ndi.binary_erosion(cell, iterations=1)
        membrane = cell & ~inner
        cell_type = float(cell_types[cell_id])
        stage = float(cell_stages[cell_id])
        expression_jitter = rng.lognormal(mean=0.0, sigma=0.20)

        markers[0, nucleus] = expression_jitter * (0.48 + 0.42 * stage)
        markers[1, membrane] = expression_jitter * (0.44 + 0.13 * (1.0 - cell_type))
        markers[1, cytoplasm] += expression_jitter * 0.08
        markers[2, cytoplasm] = expression_jitter * (
            0.27 + 0.14 * cell_type + 0.20 * (1.0 - stage)
        )
        markers[2, nucleus] = expression_jitter * (0.10 + 0.10 * cell_type)

        # ERK is deliberately related to both stage and the other marker patterns.
        # That makes held-out channel prediction a meaningful preservation test.
        markers[3, cytoplasm] = expression_jitter * (
            0.14 + 0.33 * stage + 0.08 * cell_type
        )
        markers[3, nucleus] = expression_jitter * (
            0.18 + 0.48 * stage + 0.05 * cell_type
        )

    sigmas = (0.9, 0.7, 1.0, 1.0)
    for channel, sigma in enumerate(sigmas):
        markers[channel] = ndi.gaussian_filter(markers[channel], sigma=sigma)
        texture = ndi.gaussian_filter(rng.normal(size=shape), sigma=2.0)
        texture /= np.std(texture) + 1e-6
        markers[channel] *= np.clip(1.0 + 0.025 * texture, 0.9, 1.1)

    maximum = np.quantile(markers, 0.999)
    return np.clip(markers / (maximum + 1e-8), 0.0, 1.0).astype(np.float32)


def _illumination_field(
    rng: np.random.Generator, shape: tuple[int, int], severity: float
) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    y = yy / max(height - 1, 1) - 0.5
    x = xx / max(width - 1, 1) - 0.5
    angle = rng.uniform(0.0, 2.0 * np.pi)
    gradient = np.cos(angle) * x + np.sin(angle) * y
    vignette = x**2 + y**2
    field = 1.0 + severity * (0.42 * gradient - 0.38 * vignette)
    return np.clip(field, 0.48, 1.35).astype(np.float32)


def _background_field(
    rng: np.random.Generator, shape: tuple[int, int], severity: float
) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    center_y = rng.uniform(0.2, 0.8) * height
    center_x = rng.uniform(0.2, 0.8) * width
    spread = rng.uniform(0.22, 0.38) * min(shape)
    blob = np.exp(-((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2.0 * spread**2))
    ramp = xx / max(width - 1, 1)
    return (severity * (0.025 + 0.055 * blob + 0.025 * ramp)).astype(np.float32)


def _camera_noise(
    rng: np.random.Generator, image: np.ndarray, severity: float
) -> np.ndarray:
    photons = max(35.0, 130.0 / max(severity, 0.2))
    poisson = rng.poisson(np.clip(image, 0.0, None) * photons) / photons
    read_noise = rng.normal(scale=0.008 * severity, size=image.shape)
    return np.clip(poisson + read_noise, 0.0, None).astype(np.float32)


def generate_sample(
    sample_id: int,
    image_size: int = 96,
    n_cells: int = 20,
    severity: float = 1.0,
    seed: int = 23,
) -> SyntheticSample:
    """Generate one deterministic, spatially multiplexed field of view."""

    # Biology is held fixed across severity levels, enabling paired robustness tests.
    biology_rng = np.random.default_rng(seed + 1009 * sample_id)
    artifact_rng = np.random.default_rng(seed + 1009 * sample_id + 100_003)
    labels, nuclei, cell_types, cell_stages = _draw_cells(
        biology_rng, image_size, n_cells
    )
    clean = _render_clean_markers(
        biology_rng, labels, nuclei, cell_types, cell_stages
    )

    n_cycles = len(CHANNEL_NAMES)
    observed = np.zeros((n_cycles, 2, image_size, image_size), dtype=np.float32)
    shifts = np.zeros((n_cycles, 2), dtype=np.float32)

    for cycle in range(n_cycles):
        if cycle > 0:
            shifts[cycle] = artifact_rng.uniform(-2.8 * severity, 2.8 * severity, size=2)
        illumination = _illumination_field(artifact_rng, labels.shape, severity)
        background = _background_field(artifact_rng, labels.shape, severity)
        bleaching = max(0.65, 1.0 - 0.055 * severity * cycle)

        anchor = ndi.shift(clean[0], shifts[cycle], order=1, mode="constant", cval=0.0)
        marker = ndi.shift(clean[cycle], shifts[cycle], order=1, mode="constant", cval=0.0)
        observed[cycle, 0] = _camera_noise(
            artifact_rng, bleaching * anchor * illumination + background, severity
        )
        observed[cycle, 1] = _camera_noise(
            artifact_rng, bleaching * marker * illumination + background, severity
        )

    return SyntheticSample(
        sample_id=sample_id,
        clean_markers=clean,
        observed_cycles=observed,
        cell_labels=labels,
        nucleus_labels=nuclei,
        cell_types=cell_types,
        cell_stages=cell_stages,
        acquisition_shifts=shifts,
        severity=float(severity),
    )


def generate_dataset(
    n_samples: int,
    image_size: int = 96,
    n_cells: int = 20,
    severity: float = 1.0,
    seed: int = 23,
) -> list[SyntheticSample]:
    """Generate a deterministic collection of independent fields of view."""

    return [
        generate_sample(i, image_size, n_cells, severity, seed)
        for i in range(n_samples)
    ]
