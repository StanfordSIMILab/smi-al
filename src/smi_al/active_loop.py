from pathlib import Path
import csv, json
from typing import Dict
import numpy as np
from rich import print

from smi_al.utils.common import set_seed, list_images, ensure_dir
from smi_al.io.video import read_image, to_gray
from smi_al.sampling.fvi import compute_fvi, threshold_fvi, keep_indices_above
from smi_al.embeddings.clip_embedder import embed_images
from smi_al.embeddings.umap_reduce import reduce_umap
from smi_al.sampling.cluster import elbow_k, kmeans_fit
from smi_al.sampling.select import centroid_proximal_selection

def run_phase_a(cfg: Dict):
    set_seed(cfg['selection'].get('seed', 42))
    root = Path(cfg['data']['root'])
    pattern = cfg['data'].get('glob', '*.png')
    outdir = Path(cfg['data']['output_dir'])
    ensure_dir(outdir); ensure_dir(outdir / 'figs')
    paths = list_images(root, pattern)
    print(f'[bold]Found[/bold] {len(paths)} images under {root}')

    # 1) FVI
    grays = [to_gray(read_image(p)) for p in paths]
    fvi = compute_fvi(grays, cfg['fvi']['method'])
    thr = threshold_fvi(fvi, cfg['fvi']['threshold_method'], cfg['fvi']['get']('std_k',2.0), cfg['fvi']['get']('percentile',80))
    keep_idx = keep_indices_above(fvi, thr)
    kept_paths = [paths[i] for i in keep_idx]
    print(f'FVI keep: {len(kept_paths)}/{len(paths)} images (thr={thr:.2f})')

    # 2) Embeddings
    X = embed_images(kept_paths, cfg['embedder']['model'], cfg['embedder']['pretrained'])

    # 3) UMAP
    Z = reduce_umap(X, cfg['umap']['n_neighbors'], cfg['umap']['min_dist'], cfg['umap']['dim'], cfg['selection']['seed'])

    # 4) Cluster + elbow
    k = elbow_k(Z, cfg['cluster']['k_max'], cfg['cluster']['elbow_smooth'], cfg['selection']['seed'])
    labels, centers = kmeans_fit(Z, k, cfg['selection']['seed'])
    print(f'Elbow k={k} clusters')

    # 5) Select m per cluster
    sel_idx = centroid_proximal_selection(Z, labels, centers, cfg['selection']['m_per_cluster'], cfg['selection']['metric'], cfg['selection']['seed'])
    sel_paths = [kept_paths[i] for i in sel_idx]
    print(f'Selected {len(sel_paths)} frames')

    # 6) Save manifest
    manifest = outdir / 'selected_frames.csv'
    with open(manifest, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['path'])
        for p in sel_paths: w.writerow([p])
    print(f'[green]Wrote[/green] {manifest}')
