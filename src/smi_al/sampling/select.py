from typing import Tuple, Dict
import numpy as np

def centroid_proximal_selection(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, m_per_cluster: int = 3, metric: str = 'euclidean', seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    idxs = []
    for c in range(centers.shape[0]):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        xm = X[members]
        center = centers[c][None, :]
        if metric == 'euclidean':
            d = ((xm - center)**2).sum(axis=1)**0.5
        else:
            raise ValueError('Only euclidean supported')
        order = np.argsort(d)
        take = members[order[:min(m_per_cluster, members.size)]]
        idxs.append(take)
    if not idxs:
        return np.array([], dtype=int)
    return np.concatenate(idxs, axis=0)

def nearest_centroid_index(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # for diagnostics
    d = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
    return d.argmin(axis=1)
