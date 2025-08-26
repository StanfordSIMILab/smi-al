import numpy as np
from smi_al.sampling.fvi import compute_fvi

def test_fvi_len():
    # 3 frames -> fvi length 3, fvi[0]==0
    frames = [np.zeros((10,10), np.uint8), np.ones((10,10), np.uint8)*10, np.ones((10,10), np.uint8)*20]
    fvi = compute_fvi(frames)
    assert len(fvi)==3 and fvi[0]==0
