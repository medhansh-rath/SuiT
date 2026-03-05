"""Base class for superpixel algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class AlgorithmResult:
    """Result from superpixel segmentation."""
    labels: np.ndarray  # 2D array of superpixel labels
    num_labels: int  # Number of superpixels
    runtime_seconds: float  # Wall-clock time for segmentation
    metadata: dict  # Algorithm-specific metadata


class SuperpixelAlgorithm(ABC):
    """Base class for superpixel algorithms.
    
    Each algorithm must implement segment() which takes RGB and depth images
    and returns superpixel labels.
    """
    
    def __init__(self, name: str, settings: dict):
        """Initialize algorithm with name and settings.
        
        Args:
            name: Human-readable name of the algorithm
            settings: Dictionary of algorithm-specific hyperparameters
        """
        self.name = name
        self.settings = settings.copy()
    
    @abstractmethod
    def segment(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> AlgorithmResult:
        """Segment RGB-D image into superpixels.
        
        Args:
            rgb: RGB image (H, W, 3), uint8, values 0-255
            depth: Depth image (H, W), float32, normalized to [0, 1]
            normals: (Optional) Surface normals (H, W, 3), float32, normalized
        
        Returns:
            AlgorithmResult with labels and metadata
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.name}({self.settings})"
