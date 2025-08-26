from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans

def elbow_k(X: np.ndarray, k_max: int = 32, smooth: int = 3, seed: int = 42) -> int:
    inertias = []
    ks = list(range(2, k_max+1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init='auto', random_state=seed)
        km.fit(X)
        inertias.append(km.inertia_)
    # Simple elbow: point of maximal second-difference (discrete curvature)
    y = np.array(inertias, dtype=np.float64)
    d1 = np.diff(y)
    d2 = np.diff(d1)
    idx = int(np.argmin(d2)) + 2  # offset for second diff and k starting at 2
    return ks[idx]

def kmeans_fit(X: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, n_init='auto', random_state=seed)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    return labels, centers
