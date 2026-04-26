"""Image preprocessing: blur, sharpen, contrast enhancement, flat-field correction."""

import numpy as np
import cv2


def flat_field_correct(gray: np.ndarray, flat_gray: np.ndarray) -> np.ndarray:
    """
    Divide by a normalised flat-field to remove illumination non-uniformity.

    flat_gray should be an SEM image of the glass matrix without crystals
    (or a low-crystal region). A heavy Gaussian blur extracts the slow
    illumination envelope before division so local pixel noise in the
    flat-field image doesn't get amplified.
    """
    ff = cv2.GaussianBlur(flat_gray.astype(np.float32), (0, 0), 30)
    ff_norm = ff / (ff.mean() + 1e-6)                  # normalise so mean = 1
    corrected = gray.astype(np.float32) / (ff_norm + 1e-6)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def preprocess_image(
    gray: np.ndarray,
    blur_sigma: float = 1.0,
    sharpen_strength: float = 0.0,
    clahe_clip: float = 0.0,
) -> np.ndarray:
    img = gray.astype(np.float32)

    if blur_sigma > 0.3:
        ksize = max(3, int(blur_sigma * 6) | 1)  # ensure odd
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

    if sharpen_strength > 0.05:
        blurred = cv2.GaussianBlur(img, (7, 7), 2.0)
        img = img + sharpen_strength * (img - blurred)
        img = np.clip(img, 0, 255)

    result = img.astype(np.uint8)

    if clahe_clip > 0.1:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        result = clahe.apply(result)

    return result
