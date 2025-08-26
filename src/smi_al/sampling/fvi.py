from typing import List, Tuple
import numpy as np
import cv2

def compute_fvi(gray_frames: List[np.ndarray], method: str = 'grayscale_mad') -> np.ndarray:
    """Compute Frame Variance Index over a sequence (per-frame dissimilarity to prev).
    Returns array with length == len(frames); fvi[0] = 0 by definition.
    """
    n = len(gray_frames)
    fvi = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        if method == 'grayscale_mad':
            diff = np.abs(gray_frames[i].astype(np.float32) - gray_frames[i-1].astype(np.float32))
            fvi[i] = float(diff.mean())
        else:
            raise ValueError(f'Unknown FVI method: {method}')
    return fvi

def threshold_fvi(fvi: np.ndarray, method: str = 'std', std_k: float = 2.0, percentile: float = 80.0) -> float:
    mu, sigma = float(fvi.mean()), float(fvi.std())
    if method == 'std':
        return mu + std_k * sigma
    elif method == 'percentile':
        return float(np.percentile(fvi, percentile))
    else:
        raise ValueError('method must be "std" or "percentile"')

def keep_indices_above(fvi: np.ndarray, thr: float) -> np.ndarray:
    idx = np.where(fvi >= thr)[0]
    # keep first frame to preserve context
    if 0 not in idx:
        idx = np.sort(np.r_[0, idx])
    return idx
