from pathlib import Path
import cv2
from typing import List, Tuple

def read_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def save_fig(path: str, fig):
    # expects a matplotlib Figure
    fig.savefig(path, bbox_inches='tight', dpi=160)
