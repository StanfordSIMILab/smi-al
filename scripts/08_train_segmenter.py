# Integrate your training harness here (Detectron2 / MMSegmentation / etc.).

print("""
[Train Segmenter]
- Option A: Detectron2 + Mask2Former:
  * pip install 'git+https://github.com/facebookresearch/detectron2.git'
  * Integrate your dataset in COCO format; convert CVAT-exported annotations.
  * Use a Mask2Former config and point DATASETS.TRAIN/TEST to your splits.
- Option B: MMSegmentation:
  * pip install mmsegmentation
  * Convert to mmseg formats and use a mask2former-like config.

This repo focuses on the AL selection side; training code is left up to your infra.
""")
