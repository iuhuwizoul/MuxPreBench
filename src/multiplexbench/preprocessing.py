"""Composable background, illumination, and registration methods."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from .data import SyntheticSample


@dataclass(frozen=True)
class PipelineOutput:
    markers: np.ndarray
    correction_shifts: np.ndarray
    runtime_ms: float


def subtract_background(image: np.ndarray, method: str, radius: int) -> np.ndarray:
    """Subtract a non-negative background estimate."""

    if method == "none":
        return image.astype(np.float32, copy=True)
    if method == "percentile":
        return np.clip(image - np.percentile(image, 10), 0.0, None).astype(np.float32)
    if method == "morphology":
        size = 2 * radius + 1
        estimate = ndi.grey_opening(image, size=(size, size))
        return np.clip(image - estimate, 0.0, None).astype(np.float32)
    raise ValueError(f"Unknown background method: {method}")


def correct_illumination(image: np.ndarray, method: str, sigma: float) -> np.ndarray:
    """Correct low-frequency multiplicative intensity variation."""

    if method == "none":
        return image.astype(np.float32, copy=True)
    if method == "gaussian":
        smooth = ndi.gaussian_filter(image, sigma=sigma)
        positive = smooth[smooth > 0]
        if positive.size == 0:
            return image.astype(np.float32, copy=True)
        floor = max(float(np.quantile(positive, 0.2)), 1e-4)
        field = np.maximum(smooth, floor)
        scale = float(np.median(field))
        return (image / field * scale).astype(np.float32)
    raise ValueError(f"Unknown illumination method: {method}")


class PreprocessingPipeline:
    """A small sklearn-like transformer for iterative imaging cycles."""

    def __init__(
        self,
        background: str = "none",
        illumination: str = "none",
        registration: str = "none",
        background_radius: int = 9,
        flatfield_sigma: float = 18.0,
        registration_upsample: int = 10,
    ) -> None:
        self.background = background
        self.illumination = illumination
        self.registration = registration
        self.background_radius = background_radius
        self.flatfield_sigma = flatfield_sigma
        self.registration_upsample = registration_upsample

    def _correct_intensity(self, image: np.ndarray) -> np.ndarray:
        illumination_corrected = correct_illumination(
            image, self.illumination, self.flatfield_sigma
        )
        return subtract_background(
            illumination_corrected, self.background, self.background_radius
        )

    def transform(self, sample: SyntheticSample) -> PipelineOutput:
        start = perf_counter()
        cycles = sample.observed_cycles
        anchors = np.stack([self._correct_intensity(x[0]) for x in cycles])
        markers = np.stack([self._correct_intensity(x[1]) for x in cycles])
        corrections = np.zeros((len(cycles), 2), dtype=np.float32)

        if self.registration == "phase":
            reference = anchors[0]
            for cycle in range(1, len(cycles)):
                shift, _, _ = phase_cross_correlation(
                    reference,
                    anchors[cycle],
                    upsample_factor=self.registration_upsample,
                    normalization=None,
                )
                corrections[cycle] = shift
                markers[cycle] = ndi.shift(
                    markers[cycle], shift, order=1, mode="constant", cval=0.0
                )
        elif self.registration != "none":
            raise ValueError(f"Unknown registration method: {self.registration}")

        return PipelineOutput(
            markers=markers.astype(np.float32),
            correction_shifts=corrections,
            runtime_ms=1000.0 * (perf_counter() - start),
        )
