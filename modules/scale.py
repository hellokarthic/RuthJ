"""Scale bar detection and calibration for SEM images."""

import re
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ScaleInfo:
    nm_per_pixel: float
    scale_bar_pixels: Optional[int]
    scale_bar_value_nm: Optional[float]
    ocr_text: Optional[str]
    method: str  # 'tiff_metadata' | 'px_metadata' | 'scale_bar_ocr' | 'manual' | 'fallback_23nm'
    info_bar_row: int
    scale_bar_coords: Optional[Tuple[int, int, int]] = None  # (col_start, col_end, row)


def _read_tiff_nm_per_pixel(pil_image) -> Optional[float]:
    """
    Try to read nm/pixel directly from TIFF header tags.

    Supports:
      • FEI / Thermo Fisher — tag 34682 XML blob with PixelWidth (metres)
      • Generic ImageDescription (tag 270) — plain-text Px: patterns or nested XML
    """
    if pil_image is None or not hasattr(pil_image, "tag_v2"):
        return None

    tags = pil_image.tag_v2

    # ── FEI / Thermo Fisher: tag 34682 is an XML blob ────────────────────────
    if 34682 in tags:
        try:
            import xml.etree.ElementTree as ET
            raw = tags[34682]
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")
            elif isinstance(raw, tuple):
                raw = raw[0] if raw else ""
            root = ET.fromstring(raw)
            for name in ("PixelWidth", "PixelSizeX", "pixelWidth", "pixelSizeX"):
                elem = root.find(f".//{name}")
                if elem is not None and elem.text:
                    val_m = float(elem.text.strip())
                    if 1e-12 < val_m < 1e-3:       # sanity: picometres to millimetres
                        return val_m * 1e9          # metres → nm
        except Exception:
            pass

    # ── Generic ImageDescription (tag 270) ───────────────────────────────────
    if 270 in tags:
        try:
            desc = tags[270]
            if isinstance(desc, (bytes, bytearray)):
                desc = desc.decode("utf-8", errors="ignore")
            elif isinstance(desc, tuple):
                desc = " ".join(str(x) for x in desc)

            # Plain-text Px: patterns
            nm = parse_pixel_size_nm(desc)
            if nm and 0.1 < nm < 10_000:
                return nm

            # Possible nested XML
            if "<" in desc:
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(desc)
                    for name in ("PixelWidth", "PixelSizeX", "PixelSize", "pixelSize"):
                        elem = root.find(f".//{name}")
                        if elem is not None and elem.text:
                            val = float(elem.text.strip())
                            if val < 1e-3:
                                return val * 1e9    # metres → nm
                            elif val < 1.0:
                                return val * 1000.0  # µm → nm
                            elif val < 10_000:
                                return float(val)   # already nm
                except Exception:
                    pass
        except Exception:
            pass

    return None


def detect_info_bar_row(gray: np.ndarray) -> int:
    """Find where the SEM instrument info bar starts (dark strip at bottom).

    Scans top-down inside the bottom 25% to find the FIRST sustained drop from
    bright image content into the dark info bar, avoiding the white scale bar
    line inside the bar that would confuse a bottom-up scan.
    """
    h = gray.shape[0]
    row_means = np.array([float(gray[i].mean()) for i in range(h)])

    search_start = int(h * 0.75)
    for i in range(search_start, h - 5):
        above = row_means[max(0, i - 5): i].mean()
        below = row_means[i: min(h, i + 5)].mean()
        if above > 75 and below < 65:
            return i

    return int(h * 0.93)


def detect_scale_bar_line(gray: np.ndarray, info_bar_row: int) -> Optional[Tuple[int, int, int]]:
    """Find the longest horizontal white line in the info bar.
    Returns (col_start, col_end, abs_row) or None.
    """
    info = gray[info_bar_row:, :]
    best_length = 0
    best = None

    for row_i in range(info.shape[0]):
        white_cols = np.where(info[row_i] > 230)[0]
        if len(white_cols) < 30:
            continue

        gaps = np.diff(white_cols)
        run_starts = [int(white_cols[0])]
        run_ends: list = []
        for j, g in enumerate(gaps):
            if g > 5:
                run_ends.append(int(white_cols[j]))
                run_starts.append(int(white_cols[j + 1]))
        run_ends.append(int(white_cols[-1]))

        for s, e in zip(run_starts, run_ends):
            length = e - s
            if length > best_length and length > 40:
                best_length = length
                best = (s, e, info_bar_row + row_i)

    return best


def parse_nm_from_text(text: str) -> Optional[float]:
    """Parse a scale label into nanometres. Handles µm, um, nm, mm."""
    m = re.search(r'(\d+\.?\d*)\s*[uµμ]m', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000.0
    m = re.search(r'(\d+\.?\d*)\s*nm', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d+\.?\d*)\s*mm', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e6
    return None


def parse_pixel_size_nm(text: str) -> Optional[float]:
    """Parse 'Px: 23 nm' or 'Px: 0.023 µm' style metadata."""
    m = re.search(r'[Pp]x\s*[:\s]\s*(\d+\.?\d*)\s*nm', text)
    if m:
        return float(m.group(1))
    m = re.search(r'[Pp]x\s*[:\s]\s*(\d+\.?\d*)\s*[uµμ]m', text)
    if m:
        return float(m.group(1)) * 1000.0
    return None


def auto_detect_scale(image_rgb: np.ndarray, pil_image=None) -> ScaleInfo:
    """
    Full auto-detection pipeline.
    Priority: TIFF metadata > Px: OCR metadata > scale bar + OCR > fallback.

    Parameters
    ----------
    image_rgb : RGB numpy array of the full SEM image
    pil_image : original PIL image object (optional, used for TIFF tag reading;
                must be the raw un-converted image to preserve tag_v2)
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    info_bar_row = detect_info_bar_row(gray)
    bar_coords = detect_scale_bar_line(gray, info_bar_row)

    nm_per_pixel: Optional[float] = None
    method = "fallback_23nm"
    ocr_text: Optional[str] = None
    scale_bar_nm: Optional[float] = None

    # ── Priority 1: TIFF embedded metadata (most reliable) ───────────────────
    tiff_nm = _read_tiff_nm_per_pixel(pil_image)
    if tiff_nm and 0.1 < tiff_nm < 10_000:
        nm_per_pixel = tiff_nm
        method = "tiff_metadata"

    # ── Priority 2: OCR of the info bar ──────────────────────────────────────
    if nm_per_pixel is None:
        try:
            import easyocr
            reader = easyocr.Reader(["en"], verbose=False, gpu=False)

            info_crop = image_rgb[info_bar_row:, :]
            h, w = info_crop.shape[:2]
            upscaled = cv2.resize(info_crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            results = reader.readtext(upscaled)
            ocr_text = " ".join(r[1] for r in results)

            px_nm = parse_pixel_size_nm(ocr_text)
            if px_nm and 0.5 < px_nm < 10_000:
                nm_per_pixel = px_nm
                method = "px_metadata"

            for _, text, _ in results:
                val = parse_nm_from_text(text)
                if val and 10 < val < 1e7:
                    scale_bar_nm = val
                    if nm_per_pixel is None and bar_coords is not None:
                        bar_px = bar_coords[1] - bar_coords[0]
                        if bar_px > 10:
                            nm_per_pixel = val / bar_px
                            method = "scale_bar_ocr"
                    break
        except Exception:
            pass

    if nm_per_pixel is None:
        nm_per_pixel = 23.0
        method = "fallback_23nm"

    return ScaleInfo(
        nm_per_pixel=nm_per_pixel,
        scale_bar_pixels=(bar_coords[1] - bar_coords[0]) if bar_coords else None,
        scale_bar_value_nm=scale_bar_nm,
        ocr_text=ocr_text,
        method=method,
        info_bar_row=info_bar_row,
        scale_bar_coords=bar_coords,
    )
