from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class ChromaticityMetrics:
    """
    Metrics for assessing chromaticity consistency between an original image and a processed image.

    Assumptions
    ----------
    - Images are linear RGB (or BGR if you keep it consistent for both).
    - Input arrays are shape (H, W, 3).
    - Dtype can be uint16 (0..65535) or float (typically 0..1), but both images should represent
      the same color space and be comparable.
    - These metrics do NOT use log-chromaticity (excluded as requested).

    What "chromaticity consistent" means here
    -----------------------------------------
    If the processing is a per-pixel *scalar* gain (illumination-only):
        I'(x,y) = k(x,y) * I(x,y)
    then chromaticity should be preserved and:
      - chromaticity difference should be ~0
      - angular error between RGB vectors should be ~0 degrees
    """

    eps: float = 1e-12

    # -----------------------------
    # Core computations
    # -----------------------------
    def chromaticity(self, img: np.ndarray) -> np.ndarray:
        """
        Compute normalized chromaticity per pixel: c = rgb / (r+g+b).

        Returns
        -------
        c : np.ndarray
            Shape (H, W, 3), float32.
        """
        x = img.astype(np.float32, copy=False)
        s = np.sum(x, axis=2, keepdims=True)
        s = np.maximum(s, self.eps)
        return x / s

    def chromaticity_diff(self, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """
        Per-pixel absolute chromaticity difference: |c_a - c_b|.

        Returns
        -------
        diff : np.ndarray
            Shape (H, W, 3), float32.
        """
        ca = self.chromaticity(img_a)
        cb = self.chromaticity(img_b)
        return np.abs(ca - cb).astype(np.float32)

    def chroma_l2_map(self, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """
        Per-pixel L2 norm of chromaticity difference.

        Returns
        -------
        e : np.ndarray
            Shape (H, W), float32.
        """
        d = self.chromaticity_diff(img_a, img_b)
        return np.linalg.norm(d, axis=2).astype(np.float32)

    def angular_error_deg(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        *,
        valid_mask: Optional[np.ndarray] = None,
        exclude_black: bool = True,
        black_thresh: float = 0.0,
    ) -> np.ndarray:
        """
        Compute per-pixel angular error in degrees between RGB vectors.

        Angle = arccos( (a·b) / (||a|| ||b||) )

        This is a strong test for "color direction" preservation. If b = k*a, angle is 0.

        Parameters
        ----------
        valid_mask : Optional[np.ndarray]
            Optional boolean mask (H,W). True = include.
        exclude_black : bool
            If True, excludes pixels where ||a|| or ||b|| is <= black_thresh.
        black_thresh : float
            Threshold on vector norm to treat as black.

        Returns
        -------
        ang : np.ndarray
            Per-pixel angles in degrees, shape (H, W), float32. Pixels excluded are set to NaN.
        """
        a = img_a.astype(np.float32, copy=False)
        b = img_b.astype(np.float32, copy=False)
        if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError("img_a and img_b must both be (H,W,3) and match.")

        dot = np.sum(a * b, axis=2)  # (H,W)
        na = np.linalg.norm(a, axis=2)
        nb = np.linalg.norm(b, axis=2)

        denom = (na * nb)
        denom = np.maximum(denom, self.eps)
        cos = dot / denom
        cos = np.clip(cos, -1.0, 1.0)

        ang = np.degrees(np.arccos(cos)).astype(np.float32)

        # Build exclusion mask
        m = np.ones(ang.shape, dtype=bool)
        if valid_mask is not None:
            if valid_mask.shape != ang.shape:
                raise ValueError(f"valid_mask must be (H,W). Got {valid_mask.shape}")
            m &= valid_mask.astype(bool)

        if exclude_black:
            m &= (na > black_thresh) & (nb > black_thresh)

        # set excluded pixels to NaN
        ang[~m] = np.nan
        return ang


    def angular_error_threshold_summary(self, angular_map: np.ndarray):
        """
        Return percentage of pixels under common angular thresholds.
        """
        
        thresholds = [0.25, 0.5, 1.0, 2.0]

        results = {}

        valid = np.isfinite(angular_map)
        total = np.sum(valid)

        for t in thresholds:
            results[f"pct_under_{t}_deg"] = float(
                np.sum((angular_map <= t) & valid) / total
            )

        return results
    # -----------------------------
    # Summaries
    # -----------------------------
    @staticmethod
    def _nan_summary(x: np.ndarray, percentiles: Tuple[int, ...] = (50, 90, 95, 99)) -> Dict[str, float]:
        """
        Summarize an array that may contain NaNs.
        """
        y = x[np.isfinite(x)]
        if y.size == 0:
            return {"count": 0.0}
        out: Dict[str, float] = {
            "count": float(y.size),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
        }
        for p in percentiles:
            out[f"p{p}"] = float(np.percentile(y, p))
        return out

    def evaluate(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        *,
        valid_mask: Optional[np.ndarray] = None,
        exclude_black: bool = True,
        black_thresh: float = 0.0,
        percentiles: Tuple[int, ...] = (50, 90, 95, 99),
    ) -> Dict[str, Any]:
        """
        Compute a suite of chromaticity consistency metrics.

        Returns
        -------
        results : dict
            - chroma_abs_diff: summary over |c_orig - c_proc| (all 3 channels pooled)
            - chroma_l2: summary over per-pixel L2 chroma error
            - angular_error_deg: summary over per-pixel angular error in degrees
        """
        # Chromaticity diffs
        c_diff = self.chromaticity_diff(original, processed)  # (H,W,3)
        if valid_mask is not None:
            if valid_mask.shape != c_diff.shape[:2]:
                raise ValueError("valid_mask must match (H,W) of images.")
            m3 = valid_mask[..., None].astype(bool)
            c_diff = np.where(m3, c_diff, np.nan)

        c_diff_flat = c_diff.reshape(-1, 3)
        c_diff_all = c_diff_flat[np.isfinite(c_diff_flat)]
        chroma_abs_summary = self._nan_summary(c_diff_all, percentiles=percentiles)

        # L2 map summary
        l2 = self.chroma_l2_map(original, processed)
        if valid_mask is not None:
            l2 = np.where(valid_mask.astype(bool), l2, np.nan)
        chroma_l2_summary = self._nan_summary(l2, percentiles=percentiles)

        # Angular error summary
        ang = self.angular_error_deg(
            original,
            processed,
            valid_mask=valid_mask,
            exclude_black=exclude_black,
            black_thresh=black_thresh,
        )
        
        threshold_metrics = self.angular_error_threshold_summary(ang)
        ang_summary = self._nan_summary(ang, percentiles=percentiles)

        return {
            "chroma_abs_diff": chroma_abs_summary,
            "chroma_l2": chroma_l2_summary,
            "angular_error_deg": ang_summary,
            "threshold_metrics": threshold_metrics
        }

    def error_maps(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        *,
        valid_mask: Optional[np.ndarray] = None,
        exclude_black: bool = True,
        black_thresh: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Return per-pixel error maps for visualization / debugging.

        Returns
        -------
        maps : dict
            - chroma_l2: (H,W) float32
            - angular_error_deg: (H,W) float32 with NaNs where excluded
        """
        l2 = self.chroma_l2_map(original, processed)
        ang = self.angular_error_deg(
            original,
            processed,
            valid_mask=valid_mask,
            exclude_black=exclude_black,
            black_thresh=black_thresh,
        )
        if valid_mask is not None:
            l2 = np.where(valid_mask.astype(bool), l2, np.nan)
        return {"chroma_l2": l2.astype(np.float32), "angular_error_deg": ang.astype(np.float32)}