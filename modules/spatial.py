"""Spatial analysis: nearest-neighbour distances and crystal density heatmap."""

import numpy as np
from typing import List, Dict


def nearest_neighbor_distances(measurements: List[Dict], nm_per_pixel: float) -> None:
    """
    Compute centre-to-centre nearest-neighbour distance for every crystal.
    Adds 'nearest_neighbor_um' key to each dict in-place.
    O(n²) in memory — fast for n < 10 000.
    """
    n = len(measurements)
    if n < 2:
        for m in measurements:
            m["nearest_neighbor_um"] = None
        return

    coords = np.array(
        [[m["centroid_x_px"], m["centroid_y_px"]] for m in measurements],
        dtype=float,
    )
    diff = coords[:, None, :] - coords[None, :, :]   # (n, n, 2)
    dist_px = np.sqrt((diff ** 2).sum(axis=-1))       # (n, n)
    np.fill_diagonal(dist_px, np.inf)

    nn_px = dist_px.min(axis=1)
    for m, d in zip(measurements, nn_px):
        m["nearest_neighbor_um"] = round(float(d) * nm_per_pixel / 1000.0, 4)


def density_heatmap(
    measurements: List[Dict],
    image_shape: tuple,
    nm_per_pixel: float,
    bandwidth_um: float = 2.0,
) -> np.ndarray:
    """
    Gaussian KDE crystal-density map.
    Evaluates on a 4× downsampled grid then bilinearly upsamples for speed.
    Returns float32 array of shape (H, W).
    """
    from scipy.stats import gaussian_kde
    import cv2

    h, w = image_shape[:2]
    if len(measurements) < 3:
        return np.zeros((h, w), dtype=np.float32)

    xs = np.array([m["centroid_x_px"] for m in measurements], dtype=float)
    ys = np.array([m["centroid_y_px"] for m in measurements], dtype=float)

    bw_px = max(1.0, bandwidth_um * 1000.0 / nm_per_pixel)
    bw_norm = bw_px / float(max(w, h))

    kde = gaussian_kde(np.vstack([xs, ys]), bw_method=bw_norm)

    scale = 4
    xi = np.linspace(0, w - 1, max(2, w // scale))
    yi = np.linspace(0, h - 1, max(2, h // scale))
    Xi, Yi = np.meshgrid(xi, yi)
    Z = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape).astype(np.float32)
    return cv2.resize(Z, (w, h), interpolation=cv2.INTER_LINEAR)
