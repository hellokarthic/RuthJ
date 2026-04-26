"""Crystal segmentation: thresholding, morphological cleanup, watershed."""

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_multiotsu
from skimage.morphology import white_tophat, disk
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from typing import Optional, Tuple


def _apply_threshold(gray: np.ndarray, method: str, manual_val: int) -> Tuple[np.ndarray, int]:
    if method == "Otsu":
        t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, int(t)

    if method == "Manual":
        _, binary = cv2.threshold(gray, manual_val, 255, cv2.THRESH_BINARY)
        return binary, manual_val

    if method == "Adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=-8,
        )
        return binary, -1

    if method == "Top % bright":
        t = int(np.percentile(gray, manual_val))
        _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        return binary, t

    raise ValueError(f"Unknown threshold method: {method}")


def _distance_watershed(binary_bool: np.ndarray, min_dist: int) -> np.ndarray:
    """Classic distance-transform watershed — good for well-separated crystals."""
    distance = ndi.distance_transform_edt(binary_bool)
    coords = peak_local_max(distance, min_distance=min_dist, labels=binary_bool)
    markers = np.zeros(binary_bool.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords):
        markers[r, c] = i + 1
    return watershed(-distance, markers, mask=binary_bool)


def _valley_depth(
    smoothed: np.ndarray,
    r1: int, c1: int,
    r2: int, c2: int,
    n_samples: int = 20,
) -> float:
    """
    Sample the intensity along a straight line between two seed points and
    return the minimum value (the saddle / valley between the two peaks).
    """
    rs = np.clip(np.linspace(r1, r2, n_samples).astype(int), 0, smoothed.shape[0] - 1)
    cs = np.clip(np.linspace(c1, c2, n_samples).astype(int), 0, smoothed.shape[1] - 1)
    return float(smoothed[rs, cs].min())


def _prune_seeds_by_valley(
    smoothed: np.ndarray,
    coords: np.ndarray,
    intensity_range: float,
    min_valley_frac: float,
) -> np.ndarray:
    """
    Remove seeds that do not have a sufficiently deep valley between them.

    For every pair of seeds (i, j), the depth of the valley between them is:
        depth = 0.5*(peak_i + peak_j) - saddle_ij

    If depth < min_valley_frac * intensity_range the two seeds are assumed to
    be surface-texture peaks on the *same* crystal, not separate crystals.
    The dimmer seed of the pair is discarded.

    This prevents over-segmentation of single large crystals that have
    internal BSE intensity variation (e.g. due to crystal tilt or polish).
    """
    if len(coords) < 2 or min_valley_frac <= 0:
        return coords

    keep = list(range(len(coords)))
    peak_vals = [float(smoothed[r, c]) for r, c in coords]
    min_depth = min_valley_frac * intensity_range

    changed = True
    while changed:
        changed = False
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                ii, jj = keep[i], keep[j]
                r1, c1 = coords[ii]
                r2, c2 = coords[jj]
                saddle = _valley_depth(smoothed, r1, c1, r2, c2)
                mean_peak = (peak_vals[ii] + peak_vals[jj]) / 2.0
                depth = mean_peak - saddle
                if depth < min_depth:
                    # Discard the dimmer seed
                    drop = jj if peak_vals[ii] >= peak_vals[jj] else ii
                    if drop in keep:
                        keep.remove(drop)
                        changed = True
                        break  # restart inner loops after modification
            if changed:
                break

    return coords[keep] if keep else coords[:1]


def _split_one_region(
    gray: np.ndarray,
    region_mask: np.ndarray,
    min_split_px: int,
    blur_sigma: float,
    compactness: float = 0.0,
    min_valley_frac: float = 0.10,
) -> np.ndarray:
    """
    Watershed on a single connected binary region using an intensity-based terrain.

    Terrain
    -------
    Inverted smoothed intensity + morphological gradient:
    • Crystal centres (bright peaks) → low terrain → flood origin.
    • Dark inter-crystal edges / valleys → high terrain → flood barrier.

    Over-segmentation guard
    -----------------------
    Before running watershed, seeds whose pairwise intensity valley is shallower
    than `min_valley_frac × local_intensity_range` are merged into one seed.
    This prevents surface-texture variation *within* a single crystal from being
    mistaken for a crystal boundary.
    """
    sigma = max(0.5, min_split_px / 4.0) if blur_sigma <= 0 else blur_sigma
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma).astype(np.float32)

    smoothed_masked = smoothed.copy()
    smoothed_masked[~region_mask] = 0.0

    vals = smoothed[region_mask]
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo if hi > lo else 1.0
    spacing = max(3, min_split_px)

    coords = np.empty((0, 2), dtype=int)
    for frac in (0.55, 0.40, 0.25, None):
        t = (lo + frac * span) if frac is not None else None
        raw = peak_local_max(
            smoothed_masked,
            min_distance=spacing,
            **({} if t is None else {"threshold_abs": t}),
        )
        inside = np.array([c for c in raw if region_mask[c[0], c[1]]], dtype=int)
        if len(inside) > 0:
            coords = inside
            break

    if len(coords) == 0:
        return label(region_mask)

    # ── Valley-depth pruning (over-segmentation guard) ────────────────────────
    coords = _prune_seeds_by_valley(smoothed, coords, span, min_valley_frac)

    # If only one seed survives, the whole region is one crystal — skip watershed
    if len(coords) == 1:
        return label(region_mask)

    # Dilate seeds slightly for stable flooding
    seed_u8 = np.zeros(gray.shape, dtype=np.uint8)
    for r, c in coords:
        seed_u8[r, c] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed_u8 = cv2.dilate(seed_u8, kernel, iterations=1)
    markers = label((seed_u8 > 0) & region_mask)

    # ── Watershed terrain ─────────────────────────────────────────────────────
    inv = (255.0 - smoothed.clip(0, 255)).astype(np.float32)
    mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_grad = cv2.morphologyEx(
        smoothed.clip(0, 255).astype(np.uint8), cv2.MORPH_GRADIENT, mk
    ).astype(np.float32)
    terrain = np.clip(inv * 0.6 + morph_grad * 0.4, 0, 255).astype(np.uint8)
    terrain[~region_mask] = 255

    return watershed(terrain, markers, mask=region_mask, compactness=compactness)


def _gradient_watershed(
    gray: np.ndarray,
    binary_bool: np.ndarray,
    min_dist: int,
    blur_sigma: float = 1.0,
    min_split_px: int = 10,
    compactness: float = 0.0,
    min_valley_frac: float = 0.10,
) -> np.ndarray:
    """
    Per-connected-component intensity watershed.

    Small regions (< 6 × expected single-crystal area) are kept as-is.
    Large regions are split by _split_one_region, which includes the
    valley-depth guard to prevent over-segmentation.
    """
    cc_labeled = label(binary_bool)
    result = np.zeros_like(gray, dtype=np.int32)
    next_lbl = 1

    single_crystal_area = np.pi * (min_split_px ** 2)

    for p in regionprops(cc_labeled):
        region_mask = cc_labeled == p.label

        if p.area < single_crystal_area * 6:   # conservative: 6× before splitting
            result[region_mask] = next_lbl
            next_lbl += 1
            continue

        sub = _split_one_region(
            gray, region_mask, min_split_px, blur_sigma, compactness, min_valley_frac
        )
        for sub_lbl in range(1, int(sub.max()) + 1):
            result[sub == sub_lbl] = next_lbl
            next_lbl += 1

    return result


def _sequential_remap(labeled: np.ndarray) -> np.ndarray:
    """Renumber labels 1..N without label(labeled>0) which re-merges watershed boundaries."""
    surviving = np.unique(labeled)
    surviving = surviving[surviving > 0]
    if len(surviving) == 0:
        return np.zeros_like(labeled)
    remap = np.zeros(int(labeled.max()) + 1, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(surviving, start=1):
        remap[int(old_lbl)] = new_lbl
    return remap[labeled]


def segment_crystals(
    gray: np.ndarray,
    method: str,
    manual_val: int,
    use_watershed: bool,
    min_area_px: int,
    max_area_px: int,
    splitting_mode: str = "Distance transform",
    grad_blur: float = 1.0,
    min_split_px: int = 10,
    compactness: float = 0.0,
    min_valley_frac: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Returns (labeled, binary_mask, auto_thresh).

    min_valley_frac : fraction of local intensity range that the valley between
                      two candidate seeds must be deeper than to count as a real
                      crystal boundary.  Higher → fewer splits → less over-segmentation.
    """
    binary, auto_thresh = _apply_threshold(gray, method, manual_val)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    binary_bool = binary > 0
    min_dist = max(2, int(np.sqrt(min_area_px / np.pi)))

    if use_watershed and binary_bool.any():
        if splitting_mode == "Gradient + intensity peaks":
            labeled = _gradient_watershed(
                gray, binary_bool, min_dist,
                grad_blur, min_split_px, compactness, min_valley_frac,
            )
        else:
            labeled = _distance_watershed(binary_bool, min_dist)
    else:
        labeled = label(binary_bool)

    for p in regionprops(labeled):
        if p.area < min_area_px or p.area > max_area_px:
            labeled[labeled == p.label] = 0

    return _sequential_remap(labeled), binary, (auto_thresh if auto_thresh >= 0 else None)


def segment_faint_crystals(
    gray: np.ndarray,
    bright_labeled: np.ndarray,
    min_area_px: int,
    max_area_px: int,
    tophat_radius_px: int = 30,
    sensitivity: int = 50,
    compactness: float = 0.0,
    min_valley_frac: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Detect faint / subsurface crystals below the main threshold.
    Large connected blobs are split with the same valley-depth-guarded
    watershed used for bright crystals.
    """
    radius = max(5, tophat_radius_px)
    tophat = white_tophat(gray, disk(radius)).astype(np.uint8)

    try:
        thresholds = threshold_multiotsu(tophat, classes=3)
        thresh_low  = int(thresholds[0])
        thresh_high = int(thresholds[1])
    except Exception:
        thresh_low  = int(np.percentile(tophat[tophat > 0], 30)) if tophat.any() else 10
        thresh_high = int(np.percentile(tophat[tophat > 0], 70)) if tophat.any() else 30

    nudge = int((50 - sensitivity) * thresh_low / 50) if thresh_low > 0 else 0
    effective_low = max(1, thresh_low - nudge)

    faint_mask = (tophat >= effective_low) & (tophat < thresh_high)
    faint_mask = faint_mask & (bright_labeled == 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    faint_u8 = faint_mask.astype(np.uint8) * 255
    faint_u8 = cv2.morphologyEx(faint_u8, cv2.MORPH_OPEN,  kernel, iterations=1)
    faint_u8 = cv2.morphologyEx(faint_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    faint_bool = faint_u8 > 0
    if not faint_bool.any():
        return np.zeros_like(gray, dtype=np.int32), faint_u8, effective_low, thresh_high

    min_dist = max(2, int(np.sqrt(min_area_px / np.pi)))
    single_crystal_area = np.pi * (min_dist ** 2)

    cc_faint = label(faint_bool)
    result = np.zeros_like(gray, dtype=np.int32)
    next_lbl = 1

    for p in regionprops(cc_faint):
        region_mask = cc_faint == p.label
        if p.area < single_crystal_area * 6:
            result[region_mask] = next_lbl
            next_lbl += 1
        else:
            sub = _split_one_region(
                gray, region_mask, min_dist,
                blur_sigma=1.5, compactness=compactness, min_valley_frac=min_valley_frac,
            )
            for sub_lbl in range(1, int(sub.max()) + 1):
                result[sub == sub_lbl] = next_lbl
                next_lbl += 1

    labeled_faint = result
    for p in regionprops(labeled_faint):
        if p.area < min_area_px or p.area > max_area_px:
            labeled_faint[labeled_faint == p.label] = 0

    return _sequential_remap(labeled_faint), faint_u8, effective_low, thresh_high
