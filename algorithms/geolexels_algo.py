"""GeoLexels algorithm wrapper for unified interface."""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import AlgorithmResult, SuperpixelAlgorithm

# Import GeoLexels
GEOLEXELS_DIR = Path(__file__).parent.parent.parent.parent / "GeoLexels"
if str(GEOLEXELS_DIR) not in sys.path:
    sys.path.insert(0, str(GEOLEXELS_DIR))

geolexels_segment = None
GEOLEXELS_IMPORT_ERROR = None

try:
    from GeoLexelsDemo import segment as geolexels_segment
except ImportError as e:
    GEOLEXELS_IMPORT_ERROR = e


class GeoLexelsAlgorithm(SuperpixelAlgorithm):
    """GeoLexels superpixel segmentation wrapper.
    
    Wraps the GeoLexels C++ implementation with Python interface.
    """
    
    def __init__(
        self,
        mode: int = 3,
        threshold: float = 0.25,
        focal_length: float = 1.0,
        weight_depth: float = 0.45,
        weight_normals: float = 0.1,
        doRGBtoLAB: bool = True,
        color_metric: int = 1,  # Laplace
        depth_metric: int = 1,  # Laplace
        normals_metric: int = 2,  # von Mises-Fisher
        depth_normalization_mode: int = 2,  # sensor_max
        sensor_max_depth: float = 10.0,
        trunc_laplace_cutoff: float = 0.5,
    ):
        """Initialize GeoLexels algorithm.
        
        Args:
            mode: GeoLexels clustering mode
            threshold: Superpixel merge threshold
            focal_length: Camera focal length
            weight_depth: Weight for depth features
            weight_normals: Weight for normal features
            doRGBtoLAB: Convert RGB to CIELAB
            color_metric: Color metric type (0=L2, 1=Laplace, 2=vMF)
            depth_metric: Depth metric type (0=L2, 1=Laplace, 2=vMF)
            normals_metric: Normals metric type (0=L2, 1=Laplace, 2=vMF)
            depth_normalization_mode: How to normalize depth
            sensor_max_depth: Maximum sensor depth
            trunc_laplace_cutoff: Cutoff for truncated Laplace
        """
        settings = {
            "mode": mode,
            "threshold": threshold,
            "focal_length": focal_length,
            "weight_depth": weight_depth,
            "weight_normals": weight_normals,
            "doRGBtoLAB": doRGBtoLAB,
            "color_metric": color_metric,
            "depth_metric": depth_metric,
            "normals_metric": normals_metric,
            "depth_normalization_mode": depth_normalization_mode,
            "sensor_max_depth": sensor_max_depth,
            "trunc_laplace_cutoff": trunc_laplace_cutoff,
        }
        super().__init__("GeoLexels", settings)
        self.mode = mode
        self.threshold = threshold
        self.focal_length = focal_length
        self.weight_depth = weight_depth
        self.weight_normals = weight_normals
        self.doRGBtoLAB = doRGBtoLAB
        self.color_metric = color_metric
        self.depth_metric = depth_metric
        self.normals_metric = normals_metric
        self.depth_normalization_mode = depth_normalization_mode
        self.sensor_max_depth = sensor_max_depth
        self.trunc_laplace_cutoff = trunc_laplace_cutoff
    
    def segment(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> AlgorithmResult:
        """Segment RGB-D image using GeoLexels.
        
        Note: This method requires a binary fast_cloud file as input.
        Use evaluate_algorithms.py instead, which handles the full pipeline.
        """
        raise NotImplementedError(
            "GeoLexels.segment() requires binary fast_cloud input. "
            "Use the evaluate_algorithms.py script instead."
        )
    
    def segment_from_binary(
        self,
        binary_data: np.ndarray,
        width: int,
        height: int,
    ) -> AlgorithmResult:
        """Segment using GeoLexels from fast_cloud binary data.
        
        Args:
            binary_data: Processed point cloud data from fast_cloud
            width: Image width
            height: Image height
        
        Returns:
            AlgorithmResult with superpixel labels
        """
        if geolexels_segment is None:
            raise RuntimeError(
                f"GeoLexels module not available: {GEOLEXELS_IMPORT_ERROR}. "
                "Compile GeoLexels: cd GeoLexels && python3 compile_geolexels_lib.py"
            )
        
        # Import here to avoid issues if fast_cloud binary is missing
        import tempfile
        
        t0 = time.perf_counter()
        
        # Write binary data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp_path = tmp.name
            binary_data.tofile(tmp)
        
        try:
            labels, num_labels = geolexels_segment(
                tmp_path,
                threshold=self.threshold,
                doRGBtoLAB=self.doRGBtoLAB,
                weight_depth=self.weight_depth,
                weight_normals=self.weight_normals,
                focal_length=self.focal_length,
                normals_mode=self.mode,
                is_binary=True,
                width=width,
                height=height,
                color_metric=self.color_metric,
                depth_metric=self.depth_metric,
                normals_metric=self.normals_metric,
                depth_normalization_mode=self.depth_normalization_mode,
                sensor_max_depth=float(self.sensor_max_depth),
                trunc_laplace_cutoff=self.trunc_laplace_cutoff,
            )
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        runtime = time.perf_counter() - t0
        
        return AlgorithmResult(
            labels=labels,
            num_labels=num_labels,
            runtime_seconds=runtime,
            metadata={
                "binary_width": width,
                "binary_height": height,
                "threshold": self.threshold,
                "mode": self.mode,
            },
        )
