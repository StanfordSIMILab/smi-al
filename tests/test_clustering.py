import numpy as np
from smi_al.sampling.cluster import kmeans_fit
from smi_al.sampling.select import centroid_proximal_selection

def test_select_not_empty():
    X = np.vstack([np.random.randn(20,4)+i*5 for i in range(3)])
    labels = np.array([0]*20+[1]*20+[2]*20)
    centers = np.vstack([X[labels==i].mean(0) for i in range(3)])
    idx = centroid_proximal_selection(X, labels, centers, m_per_cluster=2)
    assert idx.size == 6
