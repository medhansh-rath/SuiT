"""Superpixel algorithm implementations for comparison.

Provides unified interface for evaluating different superpixel algorithms.
"""

from .base import SuperpixelAlgorithm
from .geolexels_algo import GeoLexelsAlgorithm

__all__ = [
    "SuperpixelAlgorithm",
    "GeoLexelsAlgorithm",
]
