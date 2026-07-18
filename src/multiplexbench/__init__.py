"""Reproducible preprocessing benchmarks for multiplexed microscopy."""

from .data import CHANNEL_NAMES, SyntheticSample, generate_dataset
from .preprocessing import PreprocessingPipeline

__all__ = [
    "CHANNEL_NAMES",
    "PreprocessingPipeline",
    "SyntheticSample",
    "generate_dataset",
]
__version__ = "0.1.0"
