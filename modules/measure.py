"""Crystal measurement and statistics from labeled images."""

import numpy as np
import cv2
from skimage.measure import regionprops
from typing import List, Dict


def measure_crystals(labeled: np.ndarray, nm_per_pixel: float) -> List[Dict]:
    measurements = []
    for i, p in enumerate(regionprops(labeled)):
        area_px = p.area
        area_nm2 = area_px * (nm_per_pixel ** 2)
        diameter_nm = 2.0 * np.sqrt(area_nm2 / np.pi)

        # Axis lengths (handle skimage 0.19+ API rename)
        major = getattr(p, "axis_major_length", None) or getattr(p, "major_axis_length", 0)
        minor = getattr(p, "axis_minor_length", None) or getattr(p, "minor_axis_length", 0)
        aspect = (major / minor) if minor > 0 else 1.0

        # Max Feret diameter (skimage ≥ 0.19 rotating calipers; fallback to major axis)
        max_feret_px = getattr(p, "feret_diameter_max", None)
        max_feret_px = float(max_feret_px) if (max_feret_px is not None and float(max_feret_px) > 0) else float(major)

        # Min Feret via minimum-area bounding rectangle on the region contour
        try:
            roi = p.image.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                pts = cnts[0].squeeze()
                if pts.ndim == 1:
                    pts = pts.reshape(1, 2)
                if len(pts) >= 2:
                    rect = cv2.minAreaRect(pts.astype(np.float32))
                    min_feret_px = float(min(rect[1]))
                else:
                    min_feret_px = float(minor)
            else:
                min_feret_px = float(minor)
        except Exception:
            min_feret_px = float(minor)

        # Circularity = 4π·area / perimeter²  (1.0 = perfect circle)
        perim = float(getattr(p, "perimeter", 0))
        circularity = round(4.0 * np.pi * area_px / perim ** 2, 4) if perim > 0 else 1.0

        # Solidity = area / convex hull area
        solidity = round(float(p.solidity), 4)

        measurements.append({
            "crystal_id": i + 1,
            "diameter_um": round(diameter_nm / 1000.0, 4),
            "area_um2": round(area_nm2 / 1.0e6, 5),
            "max_feret_um": round(max_feret_px * nm_per_pixel / 1000.0, 4),
            "min_feret_um": round(min_feret_px * nm_per_pixel / 1000.0, 4),
            "major_axis_um": round(major * nm_per_pixel / 1000.0, 4),
            "minor_axis_um": round(minor * nm_per_pixel / 1000.0, 4),
            "aspect_ratio": round(aspect, 3),
            "solidity": solidity,
            "circularity": circularity,
            "centroid_x_px": round(p.centroid[1], 1),
            "centroid_y_px": round(p.centroid[0], 1),
            # internal fields — kept for stats / filtering, excluded from CSV export
            "diameter_nm": round(diameter_nm, 1),
            "diameter_um_display": round(diameter_nm / 1000.0, 3),
            "area_px": area_px,
        })
    return measurements


def compute_statistics(measurements: List[Dict]) -> Dict:
    if not measurements:
        return {}
    d = np.array([m["diameter_nm"] for m in measurements])
    nn = [m["nearest_neighbor_um"] for m in measurements if m.get("nearest_neighbor_um") is not None]
    stats: Dict = {
        "Count": int(len(d)),
        "Mean (nm)": float(np.mean(d)),
        "Median (nm)": float(np.median(d)),
        "Std Dev (nm)": float(np.std(d)),
        "Min (nm)": float(np.min(d)),
        "Max (nm)": float(np.max(d)),
        "D10 (nm)": float(np.percentile(d, 10)),
        "D50 (nm)": float(np.percentile(d, 50)),
        "D90 (nm)": float(np.percentile(d, 90)),
        "Total area (µm²)": float(np.sum([m["area_um2"] for m in measurements])),
    }
    if nn:
        stats["Mean NN dist (µm)"] = round(float(np.mean(nn)), 4)
        stats["Median NN dist (µm)"] = round(float(np.median(nn)), 4)
    return stats
