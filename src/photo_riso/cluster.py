"""K-means segmentation in CIELAB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from photo_riso.lab_colors import rgb_uint8_to_lab


@dataclass
class SegmentationResult:
    labels: np.ndarray  # HxW int, 0..k-1 by prevalence (0 = largest cluster)
    centroids_lab: np.ndarray  # kx3 CIELAB, same order as labels
    rgb: np.ndarray  # HxW3 uint8 sRGB source


def _reshape_lab(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w, _ = rgb.shape
    lab = rgb_uint8_to_lab(rgb)
    return lab.reshape(-1, 3), (h, w)


def segment_image(
    rgb: np.ndarray,
    k: int,
    sample_size: int,
    random_state: int,
    predict_chunk: int = 262_144,
) -> SegmentationResult:
    """
    Cluster pixels in CIELAB; labels sorted so index 0 is most prevalent cluster.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8")
    lab_flat, (h, w) = _reshape_lab(rgb)
    n = lab_flat.shape[0]
    rng = np.random.default_rng(random_state)
    if n <= sample_size:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=sample_size, replace=False)
    sample = lab_flat[idx]

    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    km.fit(sample)

    # Predict in horizontal strips to limit memory
    labels_flat = np.empty(n, dtype=np.int32)
    for r0 in range(0, n, predict_chunk):
        r1 = min(r0 + predict_chunk, n)
        labels_flat[r0:r1] = km.predict(lab_flat[r0:r1])

    counts = np.bincount(labels_flat, minlength=k)
    order = np.argsort(-counts)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(k)

    new_labels_flat = inverse[labels_flat.astype(np.int64)]
    centroids_old = km.cluster_centers_.astype(np.float64)
    centroids_lab = centroids_old[order]

    labels = new_labels_flat.reshape(h, w)
    return SegmentationResult(
        labels=labels,
        centroids_lab=centroids_lab,
        rgb=rgb,
    )


def pixel_centroid_distances(lab: np.ndarray, centroids_lab: np.ndarray) -> np.ndarray:
    """
    Per-pixel Euclidean distance to each centroid in CIELAB.
    Returns HxWxk float64.
    """
    d = lab[..., np.newaxis, :] - centroids_lab[np.newaxis, np.newaxis, :, :]
    return np.linalg.norm(d, axis=-1)
