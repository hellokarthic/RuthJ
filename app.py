"""SEM Crystal Size Distribution Analyzer — Streamlit app."""

import io
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from modules.scale import auto_detect_scale
from modules.preprocess import preprocess_image, flat_field_correct
from modules.segment import segment_crystals, segment_faint_crystals
from modules.measure import measure_crystals, compute_statistics
from modules.spatial import nearest_neighbor_distances, density_heatmap
from modules.stereo import saltykov_correction

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEM Crystal Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SEM Crystal Size Distribution Analyzer")
st.caption("RuO₂ crystal analysis in BSE backscattered electron images")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload SEM image (TIFF / PNG / JPG)",
    type=["tiff", "tif", "png", "jpg", "jpeg"],
)

if uploaded is None:
    st.info("Upload a BSE SEM image to begin. The tool auto-detects the scale bar from the instrument metadata strip.")
    st.stop()

# Load — keep raw PIL image for TIFF tag reading before convert() strips tags
img_bytes    = uploaded.read()
img_pil_raw  = Image.open(io.BytesIO(img_bytes))
img_pil      = img_pil_raw.convert("RGB")
img_np       = np.array(img_pil)

# ── Scale detection ───────────────────────────────────────────────────────────
with st.spinner("Detecting scale bar…"):
    scale_info = auto_detect_scale(img_np, pil_image=img_pil_raw)

img_area_rgb = img_np[: scale_info.info_bar_row, :]
img_gray     = cv2.cvtColor(img_area_rgb, cv2.COLOR_RGB2GRAY)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Scale Calibration")
    method_label = {
        "tiff_metadata": "Auto (TIFF embedded metadata)",
        "px_metadata":   "Auto (Px: metadata OCR)",
        "scale_bar_ocr": "Auto (scale bar + OCR)",
        "fallback_23nm": "Fallback default (23 nm/px)",
        "manual":        "Manual",
    }.get(scale_info.method, scale_info.method)

    st.markdown(f"**Detected:** `{scale_info.nm_per_pixel:.2f} nm/pixel`")
    st.caption(f"Method: {method_label}")
    if scale_info.scale_bar_pixels:
        st.caption(f"Scale bar line: {scale_info.scale_bar_pixels} px")
    if scale_info.ocr_text:
        with st.expander("OCR text from info bar"):
            st.code(scale_info.ocr_text[:300])

    override_scale = st.checkbox("Override scale manually")
    if override_scale:
        col_a, col_b = st.columns(2)
        with col_a:
            bar_real = st.number_input(
                "Scale bar value (nm)", 1.0, 1_000_000.0,
                float(scale_info.scale_bar_value_nm or 5000.0), 10.0,
            )
        with col_b:
            bar_px = st.number_input(
                "Scale bar pixels", 1, img_np.shape[1],
                int(scale_info.scale_bar_pixels or 213), 1,
            )
        nm_per_pixel = bar_real / bar_px
        st.success(f"→ {nm_per_pixel:.2f} nm/px")
    else:
        nm_per_pixel = scale_info.nm_per_pixel

    # ── Flat-field correction ─────────────────────────────────────────────────
    st.divider()
    st.header("Flat-Field Correction")
    use_flatfield = st.checkbox(
        "Apply flat-field correction", value=False,
        help="Upload a glass-only background image to correct for uneven BSE illumination.",
    )
    ff_gray = None
    if use_flatfield:
        ff_file = st.file_uploader(
            "Background image (glass only)", key="ff_upload",
            type=["tiff", "tif", "png", "jpg", "jpeg"],
        )
        if ff_file:
            ff_pil  = Image.open(ff_file).convert("L")
            ff_np   = np.array(ff_pil)
            ff_crop = ff_np[: min(scale_info.info_bar_row, ff_np.shape[0]), :]
            if ff_crop.shape != img_gray.shape:
                ff_crop = cv2.resize(ff_crop, (img_gray.shape[1], img_gray.shape[0]))
            ff_gray = ff_crop
            st.success("Flat-field loaded")
        else:
            st.info("Upload a background image to enable correction.")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    st.divider()
    st.header("Preprocessing")
    blur_sigma  = st.slider("Gaussian blur (σ)", 0.0, 5.0, 1.0, 0.5,
                             help="Reduces noise before thresholding")
    sharpen     = st.slider("Unsharp mask", 0.0, 3.0, 0.0, 0.1,
                             help="Enhances crystal edges")
    clahe_clip  = st.slider("CLAHE clip limit", 0.0, 10.0, 0.0, 0.5,
                             help="Adaptive contrast enhancement")

    # ── Segmentation ──────────────────────────────────────────────────────────
    st.divider()
    st.header("Segmentation")
    thresh_method = st.selectbox(
        "Threshold method",
        ["Otsu", "Manual", "Adaptive", "Top % bright"],
    )
    if thresh_method == "Manual":
        manual_val = st.slider("Threshold value (0–255)", 0, 255, 160, 1)
    elif thresh_method == "Top % bright":
        top_pct    = st.slider("Keep top % of pixels", 1, 50, 10, 1)
        manual_val = 100 - top_pct
    else:
        manual_val = 128

    use_watershed  = st.checkbox("Watershed (split touching crystals)", value=True)
    splitting_mode = st.selectbox(
        "Splitting mode",
        ["Distance transform", "Gradient + intensity peaks"],
    )
    if splitting_mode == "Gradient + intensity peaks":
        sp1, sp2 = st.columns(2)
        with sp1:
            grad_blur = st.slider("Pre-blur (σ)", 0.5, 3.0, 1.0, 0.5)
        with sp2:
            min_split_nm = st.slider("Min crystal Ø for splitting (nm)", 100, 2000, 400, 50)
        min_split_px = max(3, int(min_split_nm / 2 / nm_per_pixel))
        compactness = st.slider(
            "Boundary compactness", 0.0, 0.05, 0.0, 0.001,
            format="%.3f",
            help="Regularises watershed boundaries toward circular shapes. "
                 "0 = follow intensity edges exactly. "
                 "Increase (0.005–0.02) when boundaries between overlapping crystals are noisy.",
        )
        min_valley_frac = st.slider(
            "Min valley depth", 0.0, 0.5, 0.10, 0.01,
            format="%.2f",
            help="Only split a blob into separate crystals if the intensity valley between "
                 "two candidate peaks is deeper than this fraction of the local intensity range. "
                 "Increase (0.15–0.30) to stop single crystals being over-split. "
                 "Decrease (0.05) when crystal boundaries are very subtle.",
        )
        st.caption(f"→ seed spacing: {min_split_px} px")
    else:
        grad_blur, min_split_nm, min_split_px, compactness, min_valley_frac = 1.0, 400, 10, 0.0, 0.10

    # ── Shape filters ─────────────────────────────────────────────────────────
    st.divider()
    st.header("Shape Filters")
    sf1, sf2 = st.columns(2)
    with sf1:
        min_solidity = st.slider(
            "Min solidity", 0.0, 1.0, 0.0, 0.05,
            help="area / convex-hull area. Rejects highly concave shapes / debris. 0 = off.",
        )
    with sf2:
        min_circularity = st.slider(
            "Min circularity", 0.0, 1.0, 0.0, 0.05,
            help="4π·area / perimeter². 1 = perfect circle. Rejects fractal / rod artefacts. 0 = off.",
        )

    # ── Subsurface / low-contrast ─────────────────────────────────────────────
    st.divider()
    st.header("Subsurface / Low-Contrast Crystals")
    detect_faint = st.checkbox("Detect faint / buried crystals", value=False)
    if detect_faint:
        tophat_radius     = st.slider("Background radius (px)", 10, 80, 30, 5)
        faint_sensitivity = st.slider("Sensitivity", 0, 100, 50, 5)
    else:
        tophat_radius, faint_sensitivity = 30, 50

    # ── Size filter ───────────────────────────────────────────────────────────
    st.divider()
    st.header("Size Filter")
    sz1, sz2 = st.columns(2)
    with sz1:
        min_size_nm = st.number_input("Min diameter (nm)", 50, 5000, 200, 50)
    with sz2:
        max_size_nm = st.number_input("Max diameter (nm)", 200, 50000, 10000, 100)

    min_area_px = int((min_size_nm / nm_per_pixel / 2) ** 2 * np.pi)
    max_area_px = int((max_size_nm / nm_per_pixel / 2) ** 2 * np.pi)

# ── Processing pipeline ───────────────────────────────────────────────────────
base_name = uploaded.name.rsplit(".", 1)[0]

gray_in   = flat_field_correct(img_gray, ff_gray) if (use_flatfield and ff_gray is not None) else img_gray
processed = preprocess_image(gray_in, blur_sigma, sharpen, clahe_clip)

labeled, binary_mask, auto_thresh = segment_crystals(
    processed, thresh_method, manual_val,
    use_watershed, min_area_px, max_area_px,
    splitting_mode=splitting_mode,
    grad_blur=grad_blur,
    min_split_px=min_split_px,
    compactness=compactness,
    min_valley_frac=min_valley_frac,
)
measurements = measure_crystals(labeled, nm_per_pixel)
measurements = [m for m in measurements
                if m["solidity"] >= min_solidity and m["circularity"] >= min_circularity]
for m in measurements:
    m["type"] = "bright"

labeled_faint    = np.zeros_like(labeled)
faint_binary     = np.zeros_like(binary_mask)
faint_thresh_low = faint_thresh_high = None
measurements_faint: list = []

if detect_faint:
    labeled_faint, faint_binary, faint_thresh_low, faint_thresh_high = segment_faint_crystals(
        processed, labeled, min_area_px, max_area_px,
        tophat_radius_px=tophat_radius, sensitivity=faint_sensitivity,
        compactness=compactness, min_valley_frac=min_valley_frac,
    )
    measurements_faint = measure_crystals(labeled_faint, nm_per_pixel)
    measurements_faint = [m for m in measurements_faint
                           if m["solidity"] >= min_solidity and m["circularity"] >= min_circularity]
    for m in measurements_faint:
        m["type"] = "faint/subsurface"

all_measurements = measurements + measurements_faint
if len(all_measurements) >= 2:
    nearest_neighbor_distances(all_measurements, nm_per_pixel)

stats = compute_statistics(all_measurements) if all_measurements else {}

if detect_faint and faint_binary.any():
    composite_mask = np.zeros_like(binary_mask)
    composite_mask[binary_mask > 0]  = 255
    composite_mask[faint_binary > 0] = 140
else:
    composite_mask = binary_mask

# ── Helper functions ──────────────────────────────────────────────────────────
def _img_to_png(arr: np.ndarray) -> bytes:
    from PIL import Image as _P
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    mode = "RGB" if arr.ndim == 3 else "L"
    buf  = io.BytesIO()
    _P.fromarray(arr, mode).save(buf, format="PNG")
    return buf.getvalue()


def _add_labels(img_rgb: np.ndarray, mlist: list) -> np.ndarray:
    from PIL import Image as _P, ImageDraw, ImageFont
    out  = _P.fromarray(img_rgb.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(out)
    font = None
    for fp in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            font = ImageFont.truetype(fp, 11)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()
    for m in mlist:
        tag = str(m["crystal_id"])
        cx  = int(round(m["centroid_x_px"]))
        cy  = int(round(m["centroid_y_px"]))
        col = (255, 255, 255) if m.get("type", "bright") == "bright" else (80, 230, 255)
        bb  = draw.textbbox((0, 0), tag, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = 2
        draw.rectangle(
            [cx - tw // 2 - pad, cy - th // 2 - pad,
             cx + tw // 2 + pad, cy + th // 2 + pad],
            fill=(0, 0, 0),
        )
        draw.text((cx - tw // 2, cy - th // 2), tag, fill=col, font=font)
    return np.array(out)


def _rename(c: str) -> str:
    c = c.replace("_um2", " (µm²)").replace("_um", " (µm)").replace("_px", " (px)")
    return c.replace("_", " ")


def _style_ax(a):
    a.set_facecolor("#1a1a2e")
    for s in a.spines.values():
        s.set_edgecolor("#444")
    a.tick_params(colors="white")
    a.xaxis.label.set_color("white")
    a.yaxis.label.set_color("white")
    a.title.set_color("white")


# ── Build overlay ─────────────────────────────────────────────────────────────
n_bright = len(measurements)
n_faint  = len(measurements_faint)
overlay  = np.stack([processed, processed, processed], axis=-1).copy()

if n_bright > 0:
    diameters = np.array([m["diameter_nm"] for m in measurements])
    d_min, d_max = diameters.min(), diameters.max()
    cmap = plt.get_cmap("plasma")
    for m in measurements:
        lbl   = m["crystal_id"]
        mask  = (labeled == lbl).astype(np.uint8)
        t     = (m["diameter_nm"] - d_min) / (d_max - d_min) if d_max > d_min else 0.5
        r, g, b, _ = cmap(t)
        color = (int(b * 255), int(g * 255), int(r * 255))
        colored = np.zeros_like(overlay)
        colored[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.35, 0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color, 1)

if detect_faint and n_faint > 0:
    CYAN = (255, 230, 80)
    for m in measurements_faint:
        lbl   = m["crystal_id"]
        mask  = (labeled_faint == lbl).astype(np.uint8)
        colored = np.zeros_like(overlay)
        colored[mask > 0] = CYAN
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.25, 0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, CYAN, 2)

overlay_labeled = _add_labels(overlay, all_measurements)

# ── Coloured segmented mask ───────────────────────────────────────────────────
binary_segmented = np.zeros((*composite_mask.shape, 3), dtype=np.uint8)
binary_segmented[composite_mask == 140] = (50, 50, 50)
_seg_cmap = plt.get_cmap("tab20")
_all_seg  = measurements + measurements_faint
for idx, m in enumerate(_all_seg):
    is_faint = m.get("type") == "faint/subsurface"
    lbl_arr  = labeled_faint if is_faint else labeled
    region   = lbl_arr == m["crystal_id"]
    r, g, b, _ = _seg_cmap((idx % 20) / 20.0)
    binary_segmented[region] = (int(r * 220), int(g * 220), int(b * 220))
    mu8 = region.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary_segmented, cnts, -1, (255, 255, 255), 1)

binary_segmented_labeled = _add_labels(binary_segmented, _all_seg)

# ── 3-column display ──────────────────────────────────────────────────────────
col_orig, col_overlay, col_binary = st.columns(3)

with col_orig:
    st.subheader("Original")
    st.image(img_area_rgb, use_container_width=True)
    h_um = img_area_rgb.shape[0] * nm_per_pixel / 1000
    w_um = img_area_rgb.shape[1] * nm_per_pixel / 1000
    st.caption(f"Image area: {w_um:.1f} × {h_um:.1f} µm  |  {img_area_rgb.shape[1]} × {img_area_rgb.shape[0]} px")

with col_overlay:
    title = f"Bright: {n_bright}" + (f"  |  Faint: {n_faint}" if detect_faint else "")
    st.subheader(title)
    st.image(overlay, use_container_width=True)
    legend = "Plasma = bright (purple=small → yellow=large)"
    if detect_faint:
        legend += "  |  Cyan = faint/subsurface"
    st.caption(legend)

with col_binary:
    st.subheader("Segmented Mask")
    st.image(composite_mask, use_container_width=True)
    cap = f"Bright threshold = {auto_thresh}"
    if detect_faint and faint_thresh_low is not None:
        cap += f"  |  Faint range = {faint_thresh_low}–{faint_thresh_high}"
    st.caption(cap)

# ── Results row ───────────────────────────────────────────────────────────────
st.divider()
res_left, res_right = st.columns([1, 1])

with res_left:
    st.subheader("Size Distribution")
    if not measurements:
        st.warning("No crystals detected. Try lowering the minimum size or adjusting the threshold.")
    else:
        diameters_um       = [m["diameter_um"] for m in measurements]
        diameters_faint_um = [m["diameter_um"] for m in measurements_faint]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#0f0f13")

        ax = axes[0]
        bins = min(30, max(5, len(diameters_um) // 3))
        ax.hist(diameters_um, bins=bins, color="#7c3aed", edgecolor="white",
                linewidth=0.4, alpha=0.85, label=f"Bright (n={len(diameters_um)})")
        if detect_faint and diameters_faint_um:
            ax.hist(diameters_faint_um, bins=bins, color="#22d3ee", edgecolor="white",
                    linewidth=0.4, alpha=0.7, label=f"Faint (n={len(diameters_faint_um)})")
        ax.axvline(stats["Mean (nm)"] / 1000, color="#f59e0b", lw=1.5, linestyle="--",
                   label=f"Mean {stats['Mean (nm)'] / 1000:.3f} µm")
        ax.axvline(stats["D50 (nm)"] / 1000, color="#10b981", lw=1.5, linestyle=":",
                   label=f"D50 {stats['D50 (nm)'] / 1000:.3f} µm")
        ax.set_xlabel("Equivalent Diameter (µm)")
        ax.set_ylabel("Count")
        ax.set_title("Crystal Size Histogram")
        _style_ax(ax)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

        ax2 = axes[1]
        sorted_d_um = np.sort(diameters_um)
        cdf = np.arange(1, len(sorted_d_um) + 1) / len(sorted_d_um) * 100
        ax2.plot(sorted_d_um, cdf, color="#7c3aed", lw=2, label="Bright")
        if detect_faint and diameters_faint_um:
            sf = np.sort(diameters_faint_um)
            cf = np.arange(1, len(sf) + 1) / len(sf) * 100
            ax2.plot(sf, cf, color="#22d3ee", lw=2, linestyle="--", label="Faint")
        for pct, val_nm in [("D10", stats["D10 (nm)"]),
                              ("D50", stats["D50 (nm)"]),
                              ("D90", stats["D90 (nm)"])]:
            v = val_nm / 1000
            ax2.axvline(v, lw=1, linestyle="--", color="#a78bfa", alpha=0.7)
            ax2.text(v + 0.005, 5, pct, color="#a78bfa", fontsize=7)
        ax2.set_xlabel("Equivalent Diameter (µm)")
        ax2.set_ylabel("Cumulative %")
        ax2.set_title("Cumulative Distribution")
        _style_ax(ax2)
        if detect_faint and diameters_faint_um:
            ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with res_right:
    st.subheader("Statistics")
    if stats:
        stat_rows = []
        for k, v in stats.items():
            if "(nm)" in k:
                display = f"{v / 1000:.3f} µm"
                label   = k.replace("(nm)", "(µm)")
            elif "µm²" in k:
                display, label = f"{v:.4f} µm²", k
            elif "dist (µm)" in k:
                display, label = f"{v:.3f} µm", k
            elif k == "Count":
                display, label = str(int(v)), k
            else:
                display, label = f"{v:.3f}", k
            stat_rows.append({"Metric": label, "Value": display})
        st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)

    # CSV
    CSV_COLS = [
        "crystal_id", "type",
        "diameter_um", "area_um2",
        "max_feret_um", "min_feret_um",
        "major_axis_um", "minor_axis_um",
        "aspect_ratio", "solidity", "circularity",
        "nearest_neighbor_um",
        "centroid_x_px", "centroid_y_px",
    ]
    if all_measurements:
        df_export = pd.DataFrame(all_measurements)
        df_export = df_export[[c for c in CSV_COLS if c in df_export.columns]]
        df_export.columns = [_rename(c) for c in df_export.columns]
        csv_bytes = df_export.to_csv(index=False).encode()

        st.download_button(
            "⬇ Download measurements CSV",
            data=csv_bytes,
            file_name=f"{base_name}_crystals.csv",
            mime="text/csv",
            use_container_width=True,
        )

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("⬇ Overlay image",
                               data=_img_to_png(overlay),
                               file_name=f"{base_name}_overlay.png",
                               mime="image/png", use_container_width=True)
        with dl2:
            st.download_button("⬇ Segmented mask",
                               data=_img_to_png(composite_mask),
                               file_name=f"{base_name}_mask.png",
                               mime="image/png", use_container_width=True)
        with dl3:
            st.download_button("⬇ Original (cropped)",
                               data=_img_to_png(img_area_rgb),
                               file_name=f"{base_name}_original.png",
                               mime="image/png", use_container_width=True)

        with st.expander("View all crystal measurements"):
            st.dataframe(df_export, use_container_width=True)

# ── Size-class breakdown ──────────────────────────────────────────────────────
if all_measurements:
    with st.expander("Size-Class Breakdown"):
        bc1, bc2 = st.columns(2)
        with bc1:
            fine_cut = st.number_input("Fine/medium cutoff (µm)", 0.1, 10.0, 0.5, 0.1, key="fine_cut")
        with bc2:
            coarse_cut = st.number_input("Medium/coarse cutoff (µm)", 0.5, 20.0, 2.0, 0.1, key="coarse_cut")

        sc_classes: dict = {"Fine": [], "Medium": [], "Coarse": []}
        for m in all_measurements:
            d = m["diameter_um"]
            if d < fine_cut:
                sc_classes["Fine"].append(m)
            elif d < coarse_cut:
                sc_classes["Medium"].append(m)
            else:
                sc_classes["Coarse"].append(m)

        total_cnt  = len(all_measurements) or 1
        total_area = sum(m["area_um2"] for m in all_measurements) or 1.0
        sc_rows = []
        for cls, mlist in sc_classes.items():
            cnt   = len(mlist)
            afrac = sum(m["area_um2"] for m in mlist) / total_area * 100
            meand = float(np.mean([m["diameter_um"] for m in mlist])) if mlist else 0.0
            sc_rows.append({
                "Class": cls,
                "Range (µm)": (
                    f"< {fine_cut}" if cls == "Fine"
                    else f"{fine_cut}–{coarse_cut}" if cls == "Medium"
                    else f"> {coarse_cut}"
                ),
                "Count": cnt,
                "% Count": f"{cnt / total_cnt * 100:.1f}%",
                "Area fraction (%)": f"{afrac:.1f}%",
                "Mean Ø (µm)": f"{meand:.3f}",
            })
        st.dataframe(pd.DataFrame(sc_rows), hide_index=True, use_container_width=True)

# ── Spatial Analysis ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Spatial Analysis")
sp_left, sp_right = st.columns(2)

with sp_left:
    st.markdown("**Nearest-Neighbour Distance Distribution**")
    nn_vals = [m["nearest_neighbor_um"] for m in all_measurements
               if m.get("nearest_neighbor_um") is not None]
    if len(nn_vals) >= 2:
        fig_nn, ax_nn = plt.subplots(figsize=(5, 3))
        fig_nn.patch.set_facecolor("#0f0f13")
        _style_ax(ax_nn)
        nn_bins = min(30, max(5, len(nn_vals) // 3))
        ax_nn.hist(nn_vals, bins=nn_bins, color="#22d3ee",
                   edgecolor="white", linewidth=0.4, alpha=0.85)
        mean_nn = float(np.mean(nn_vals))
        ax_nn.axvline(mean_nn, color="#f59e0b", lw=1.5, linestyle="--",
                      label=f"Mean {mean_nn:.3f} µm")
        ax_nn.set_xlabel("Nearest-neighbour distance (µm)")
        ax_nn.set_ylabel("Count")
        ax_nn.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
        fig_nn.tight_layout()
        st.pyplot(fig_nn)
        plt.close(fig_nn)
    else:
        st.info("Need ≥ 2 crystals for nearest-neighbour analysis.")

with sp_right:
    st.markdown("**Crystal Density Heatmap**")
    enable_heatmap = st.checkbox(
        "Compute density heatmap", value=False,
        help="Gaussian KDE of crystal centroid positions overlaid on the image.",
    )
    if enable_heatmap and len(all_measurements) >= 3:
        bw_um = st.slider("Bandwidth (µm)", 0.5, 10.0, 2.0, 0.5)
        Z = density_heatmap(all_measurements, img_area_rgb.shape, nm_per_pixel, bw_um)
        fig_hm, ax_hm = plt.subplots(figsize=(5, 4))
        fig_hm.patch.set_facecolor("#0f0f13")
        ax_hm.imshow(img_area_rgb, cmap="gray", aspect="auto")
        ax_hm.imshow(Z, cmap="hot", alpha=0.55, aspect="auto")
        ax_hm.axis("off")
        ax_hm.set_title("Crystal density (KDE)", color="white", pad=4)
        fig_hm.tight_layout()
        st.pyplot(fig_hm)
        plt.close(fig_hm)
    elif enable_heatmap:
        st.info("Need ≥ 3 crystals for heatmap.")

# ── Stereological Correction (Saltykov) ───────────────────────────────────────
st.divider()
st.subheader("Stereological Correction — Saltykov Method")
st.caption(
    "Estimates the true 3D sphere-diameter distribution from 2D cross-section "
    "measurements. Assumes spherical particles and a random sectioning plane."
)
enable_saltykov = st.checkbox("Show Saltykov correction", value=False)
if enable_saltykov:
    if len(measurements) < 6:
        st.warning("Need ≥ 6 bright crystals for Saltykov correction.")
    else:
        n_sal     = st.slider("Number of size classes", 5, 20, 10, 1)
        d_nm_arr  = np.array([m["diameter_nm"] for m in measurements])
        ctr_nm, N_V, bins_nm = saltykov_correction(d_nm_arr, n_bins=n_sal)
        ctr_um  = ctr_nm / 1000.0
        bins_um = bins_nm / 1000.0
        bar_w   = (bins_um[1] - bins_um[0]) * 0.8

        N_A_counts, _ = np.histogram(d_nm_arr / 1000.0, bins=bins_um)
        N_A_rel = N_A_counts.astype(float)
        if N_A_rel.sum() > 0:
            N_A_rel /= N_A_rel.sum()

        fig_s, axes_s = plt.subplots(1, 2, figsize=(10, 4))
        fig_s.patch.set_facecolor("#0f0f13")
        for ax in axes_s:
            _style_ax(ax)

        axes_s[0].bar(ctr_um, N_A_rel, width=bar_w, color="#7c3aed",
                      edgecolor="white", linewidth=0.4, alpha=0.85)
        axes_s[0].set_xlabel("Diameter (µm)")
        axes_s[0].set_ylabel("Relative frequency")
        axes_s[0].set_title("2D Measured (apparent)")

        axes_s[1].bar(ctr_um, N_V, width=bar_w, color="#f59e0b",
                      edgecolor="white", linewidth=0.4, alpha=0.85)
        axes_s[1].set_xlabel("Diameter (µm)")
        axes_s[1].set_ylabel("Relative frequency")
        axes_s[1].set_title("3D Corrected (Saltykov)")

        fig_s.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)

# ── Crystal ID Map ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Crystal ID Map")

lbl_ov_col, lbl_mask_col = st.columns(2)
with lbl_ov_col:
    st.image(overlay_labeled, use_container_width=True,
             caption="Overlay — each crystal numbered at its centroid")
    st.download_button("⬇ Labeled overlay",
                       data=_img_to_png(overlay_labeled),
                       file_name=f"{base_name}_labeled_overlay.png",
                       mime="image/png", use_container_width=True)
with lbl_mask_col:
    st.image(binary_segmented_labeled, use_container_width=True,
             caption="Segmented mask — each crystal coloured and numbered")
    st.download_button("⬇ Labeled segmented mask",
                       data=_img_to_png(binary_segmented_labeled),
                       file_name=f"{base_name}_segmented_mask_labeled.png",
                       mime="image/png", use_container_width=True)

if all_measurements:
    tbl_rows = []
    for m in all_measurements:
        tbl_rows.append({
            "ID":             m["crystal_id"],
            "Type":           m.get("type", "bright"),
            "Ø (µm)":         f"{m['diameter_um']:.3f}",
            "Max Feret (µm)": f"{m['max_feret_um']:.3f}",
            "Min Feret (µm)": f"{m['min_feret_um']:.3f}",
            "Area (µm²)":     f"{m['area_um2']:.4f}",
            "Solidity":       f"{m['solidity']:.3f}",
            "Circularity":    f"{m['circularity']:.3f}",
            "NN dist (µm)":   f"{m['nearest_neighbor_um']:.3f}" if m.get("nearest_neighbor_um") else "—",
        })
    st.dataframe(pd.DataFrame(tbl_rows).set_index("ID"),
                 use_container_width=True, height=400)

# ── Manual correction ─────────────────────────────────────────────────────────
with st.expander("Manual correction — remove false positives"):
    st.caption("Enter crystal IDs (from the labeled overlay above) that are artefacts or mis-detections.")
    rm_str = st.text_input("Crystal IDs to remove (comma-separated)", placeholder="e.g. 3, 7, 15")
    if rm_str.strip():
        try:
            rm_ids   = {int(x.strip()) for x in rm_str.split(",") if x.strip()}
            filtered = [m for m in all_measurements if m["crystal_id"] not in rm_ids]
            st.success(f"Removed {len(all_measurements) - len(filtered)} crystal(s). Remaining: {len(filtered)}")
            if filtered:
                corr_stats = compute_statistics(filtered)
                corr_rows  = []
                for k, v in corr_stats.items():
                    if "(nm)" in k:
                        display, label = f"{v / 1000:.3f} µm", k.replace("(nm)", "(µm)")
                    elif "µm²" in k:
                        display, label = f"{v:.4f} µm²", k
                    elif "dist (µm)" in k:
                        display, label = f"{v:.3f} µm", k
                    elif k == "Count":
                        display, label = str(int(v)), k
                    else:
                        display, label = f"{v:.3f}", k
                    corr_rows.append({"Metric": label, "Value": display})
                st.dataframe(pd.DataFrame(corr_rows), hide_index=True, use_container_width=True)

                df_corr = pd.DataFrame(filtered)
                df_corr = df_corr[[c for c in CSV_COLS if c in df_corr.columns]]
                df_corr.columns = [_rename(c) for c in df_corr.columns]
                st.download_button("⬇ Download corrected CSV",
                                   data=df_corr.to_csv(index=False).encode(),
                                   file_name=f"{base_name}_corrected.csv",
                                   mime="text/csv", use_container_width=True)
        except ValueError:
            st.error("Invalid format. Enter comma-separated integers, e.g.: 3, 7, 15")

# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Batch Processing")
st.caption("Process multiple SEM images with the same sidebar settings and export a combined CSV.")

batch_files = st.file_uploader(
    "Upload SEM images for batch analysis",
    type=["tiff", "tif", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="batch_uploader",
)

if batch_files and st.button("▶ Run Batch Analysis", type="primary"):
    all_batch_meas: list = []
    batch_summary:  list = []

    prog = st.progress(0, text="Starting…")
    for idx, bf in enumerate(batch_files):
        prog.progress((idx + 1) / len(batch_files), text=f"Processing {bf.name}…")
        try:
            bf_bytes  = bf.read()
            bf_raw    = Image.open(io.BytesIO(bf_bytes))
            bf_rgb    = np.array(bf_raw.convert("RGB"))
            bf_scale  = auto_detect_scale(bf_rgb, pil_image=bf_raw)
            bf_nm_px  = bf_scale.nm_per_pixel
            bf_area   = bf_rgb[: bf_scale.info_bar_row, :]
            bf_gray   = cv2.cvtColor(bf_area, cv2.COLOR_RGB2GRAY)

            bf_min_area = int((min_size_nm / bf_nm_px / 2) ** 2 * np.pi)
            bf_max_area = int((max_size_nm / bf_nm_px / 2) ** 2 * np.pi)
            bf_min_split_px = max(3, int(min_split_nm / 2 / bf_nm_px)) if splitting_mode == "Gradient + intensity peaks" else min_split_px

            bf_proc = preprocess_image(bf_gray, blur_sigma, sharpen, clahe_clip)
            bf_lbl, _, _ = segment_crystals(
                bf_proc, thresh_method, manual_val,
                use_watershed, bf_min_area, bf_max_area,
                splitting_mode=splitting_mode,
                grad_blur=grad_blur,
                min_split_px=bf_min_split_px,
                compactness=compactness,
                min_valley_frac=min_valley_frac,
            )
            bf_meas = measure_crystals(bf_lbl, bf_nm_px)
            bf_meas = [m for m in bf_meas
                       if m["solidity"] >= min_solidity and m["circularity"] >= min_circularity]
            for m in bf_meas:
                m["type"]        = "bright"
                m["source_file"] = bf.name
            if len(bf_meas) >= 2:
                nearest_neighbor_distances(bf_meas, bf_nm_px)

            all_batch_meas.extend(bf_meas)
            bf_st = compute_statistics(bf_meas)
            if bf_st:
                batch_summary.append({
                    "File":        bf.name,
                    "Count":       bf_st["Count"],
                    "Mean (µm)":   f"{bf_st['Mean (nm)'] / 1000:.3f}",
                    "Median (µm)": f"{bf_st['Median (nm)'] / 1000:.3f}",
                    "D10 (µm)":    f"{bf_st['D10 (nm)'] / 1000:.3f}",
                    "D90 (µm)":    f"{bf_st['D90 (nm)'] / 1000:.3f}",
                    "nm/px":       f"{bf_nm_px:.2f}",
                })
        except Exception as exc:
            st.warning(f"Skipped {bf.name}: {exc}")

    prog.empty()

    if batch_summary:
        st.markdown("**Per-image summary**")
        st.dataframe(pd.DataFrame(batch_summary), hide_index=True, use_container_width=True)

    if all_batch_meas:
        all_d_um = [m["diameter_um"] for m in all_batch_meas]
        fig_b, ax_b = plt.subplots(figsize=(9, 3))
        fig_b.patch.set_facecolor("#0f0f13")
        _style_ax(ax_b)
        ax_b.hist(all_d_um, bins=40, color="#7c3aed", edgecolor="white",
                  linewidth=0.3, alpha=0.85)
        ax_b.set_xlabel("Equivalent Diameter (µm)")
        ax_b.set_ylabel("Count")
        ax_b.set_title(
            f"Combined — {len(all_batch_meas)} crystals across {len(batch_files)} images",
            color="white",
        )
        fig_b.tight_layout()
        st.pyplot(fig_b)
        plt.close(fig_b)

        BATCH_COLS = [
            "source_file", "crystal_id", "type",
            "diameter_um", "area_um2",
            "max_feret_um", "min_feret_um",
            "major_axis_um", "minor_axis_um",
            "aspect_ratio", "solidity", "circularity",
            "nearest_neighbor_um",
            "centroid_x_px", "centroid_y_px",
        ]
        df_b = pd.DataFrame(all_batch_meas)
        df_b = df_b[[c for c in BATCH_COLS if c in df_b.columns]]
        df_b.columns = [_rename(c) for c in df_b.columns]
        st.download_button(
            "⬇ Download combined batch CSV",
            data=df_b.to_csv(index=False).encode(),
            file_name="batch_crystals.csv",
            mime="text/csv",
            use_container_width=True,
        )
