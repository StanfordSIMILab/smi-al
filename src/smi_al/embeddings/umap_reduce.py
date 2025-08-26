from typing import Optional
import numpy as np

def reduce_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, dim: int = 32, seed: int = 42) -> np.ndarray:
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=dim, random_state=seed)
    Z = reducer.fit_transform(X)
    return Z.astype(np.float32)
