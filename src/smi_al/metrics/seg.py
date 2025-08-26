import numpy as np

def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred & gt).sum()
    return (2*inter) / (pred.sum() + gt.sum() + eps)

def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / (union + eps)
