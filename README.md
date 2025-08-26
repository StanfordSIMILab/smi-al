# SMI Active Learning Pipeline (Phase A)

**Goal:** A practical, coverage-first active learning (AL) pipeline for surgical instance segmentation that reduces expert annotation time via **FVI filtering → embeddings → UMAP → k-means → centroid selection**, with clean interfaces to extend toward calibrated uncertainty, cost-aware batching, surgeon-in-the-loop UI, and federated deployments.

> This repository implements the **Phase A** pipeline described in the manuscript: cheap, cold-start–robust, redundancy-aware selection for neurosurgical video. It also scaffolds hooks for later phases.

## Features
- **Frame Variance Index (FVI)**: filter near-duplicate frames while retaining informative transitions.
- **Embeddings adapter**: pluggable encoders (default: OpenCLIP if available; fallback: HOG/color hist).
- **UMAP reduction**: dimensionality reduction prior to clustering.
- **k-means + elbow selection**: approximate the visual manifold and pick centroid-proximal frames.
- **Batch builder**: balanced sampling with per-cluster quota `m`.
- **Cost-aware hooks**: simple time predictors & greedy knapsack (optional).
- **Metrics**: basic mIoU/Dice + efficiency curves (minutes-normalized gains).
- **Scripts**: stepwise CLI (`00_`…`12_`) plus a `pipeline.py` orchestration.
- **Docs & configs**: ready-to-run configs for Phase A over a toy dataset; placeholders for OR-tuned Phase B/C.

## Quickstart
### 0) Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional: for CLIP embeddings
pip install open_clip_torch torch torchvision --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA/CPU
```
> If `open_clip_torch` is unavailable, the embedder falls back to a deterministic HOG+color feature.

### 1) Prepare (toy) data
Use the synthetic generator to create a tiny image set:
```bash
python scripts/gen_synthetic.py --out data/toy --n 200
```

### 2) Run Phase A (one shot)
```bash
python scripts/pipeline.py --config configs/sampling_phaseA.yaml
```
This will:
1. Compute FVI and filter frames.
2. Encode frames (OpenCLIP if available; else fallback).
3. Reduce with UMAP.
4. k-means with elbow selection.
5. Choose `m` centroid-proximal frames per cluster.
6. Export a manifest (`outputs/phaseA/selected_frames.csv`) for annotation in CVAT.
7. Plot diagnostics in `outputs/phaseA/figs/`.

### 3) Train your segmenter (placeholder)
Training Mask2Former is repo/user-specific. We provide a stub in `scripts/08_train_segmenter.py` that prints integration tips for Detectron2/Mask2Former or MMSegmentation. Replace with your training harness.

## Repo layout
```
smi-al-pipeline/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ LICENSE
├─ CITATION.cff
├─ configs/
├─ scripts/
├─ src/smi_al/
│  ├─ embeddings/
│  ├─ sampling/
│  ├─ propagation/
│  ├─ metrics/
│  ├─ io/
│  └─ utils/
├─ tests/
├─ docs/
└─ .github/workflows/
```

## Configuration
Edit `configs/sampling_phaseA.yaml`. Key fields:
- `data.root`: image root (single directory of frames).
- `fvi.threshold_method`: `"std"` or `"percentile"`, with parameters.
- `embedder.name`: `"openclip"` or `"fallback"`
- `umap`: n_neighbors, min_dist, dim
- `cluster`: k_max, elbow_smooth
- `selection.m_per_cluster`: per-cluster quota

## Roadmap (Phase B/C hooks)
- **Calibrated hybrids**: uncertainty gated within clusters; expected-model-change surrogates.
- **Cost-aware batching**: utility-per-minute via time prediction + knapsack.
- **Surgeon-in-the-loop UI**: prompts, veto, preview, time forecasts.
- **Shift & OOD**: stratified acquisition, novelty detection, open-set channel.
- **Federated AL**: local scoring + global diversity planning (indices only).

## Citing
See `CITATION.cff`. Please also cite OpenCLIP/UMAP/scikit-learn if you use those components.

---
*This is research code; use with anonymized/synthetic data unless you have approvals. Do not commit PHI/PII.*
