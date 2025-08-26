from pathlib import Path
from typing import List
import random, numpy as np

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def list_images(root: Path, pattern: str = '*.png'):
    return sorted([str(p) for p in Path(root).rglob(pattern)])

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
