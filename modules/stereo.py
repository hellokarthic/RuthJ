"""Saltykov stereological correction: 2D cross-section diameters → 3D sphere distribution."""

import numpy as np
from typing import Tuple


def saltykov_correction(
    diameters_nm: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Saltykov unfolding: estimate the true 3D sphere-diameter distribution from
    2D apparent circle diameters measured on a polished cross-section.

    A sphere of diameter D_j cut by a random plane produces a circular cross-
    section with diameter d ≤ D_j.  The Saltykov matrix encodes the probability
    that each 3D class contributes to each 2D class; the system is solved by
    back-substitution from the largest class downward.

    Parameters
    ----------
    diameters_nm : 1-D array of 2D apparent diameters (nm)
    n_bins       : number of equal-width size classes (capped at len//2)

    Returns
    -------
    centers_nm : ndarray — bin-centre diameters (nm)
    N_V        : ndarray — relative 3D frequency (sums to 1)
    bins_nm    : ndarray — bin edges (nm)
    """
    n = min(n_bins, max(3, len(diameters_nm) // 2))

    d_max = float(np.max(diameters_nm))
    bins = np.linspace(0.0, d_max, n + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0

    N_A, _ = np.histogram(diameters_nm, bins=bins)
    N_A = N_A.astype(float)

    # α(i, j) where i, j are 1-based class indices (i ≤ j):
    # probability that a sphere in 3D class j appears as a 2D circle in class i.
    # α(i,j) = √(j² − (i−1)²) − √(j² − i²)   (for unit bin width ΔD = 1)
    alpha = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            alpha[i - 1, j - 1] = (
                np.sqrt(float(j ** 2 - (i - 1) ** 2))
                - np.sqrt(float(j ** 2 - i ** 2))
            )

    # Back-substitute from the largest class downward
    N_V = np.zeros(n)
    for j in range(n - 1, -1, -1):
        corr = sum(alpha[i, j] * N_V[i] for i in range(j + 1, n))
        if alpha[j, j] > 0:
            N_V[j] = max(0.0, (N_A[j] - corr) / alpha[j, j])

    total = N_V.sum()
    if total > 0:
        N_V /= total

    return centers, N_V, bins
