"""OpenCV forensic operators that make GAN artefacts visible.

These are used for exploratory analysis and to explain predictions. The classifier itself
is trained on RGB pixels; the operators are not model inputs.
"""

from __future__ import annotations

import cv2
import numpy as np


def ela(img_bgr: np.ndarray, quality: int = 90, scale: float = 15) -> np.ndarray:
    """Error Level Analysis: re-encode as JPEG at ``quality``, subtract and amplify.

    GAN faces show uniformly low ELA (no compression history); real photos show locally
    varying ELA from earlier JPEG cycles.
    """
    ok, enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG re-encoding failed")
    rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img_bgr, rec)
    return cv2.convertScaleAbs(diff, alpha=scale)


def ela_energy(img_bgr: np.ndarray, quality: int = 90) -> float:
    """Mean ELA value: a single-number summary used in the EDA histogram."""
    return float(np.mean(ela(img_bgr, quality=quality)))


def _gray_float(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def dct_spectrum(img_bgr: np.ndarray) -> np.ndarray:
    """Log-magnitude 2D DCT. StyleGAN's transposed convolutions leave a grid in the high frequencies.

    ``cv2.dct`` needs even dimensions, so odd sizes are trimmed by one pixel.
    """
    gray = _gray_float(img_bgr)
    h, w = gray.shape
    gray = gray[: h - (h % 2), : w - (w % 2)]
    return np.log1p(np.abs(cv2.dct(gray)))


def fft_spectrum(img_bgr: np.ndarray) -> np.ndarray:
    """Centred log-magnitude FFT; radial energy distribution differs between real and fake."""
    f = np.fft.fftshift(np.fft.fft2(_gray_float(img_bgr)))
    return np.log1p(np.abs(f))


def laplacian_residual(img_bgr: np.ndarray) -> np.ndarray:
    """High-pass Laplacian residual that surfaces synthesis seams."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))


def sobel_edges(img_bgr: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


OPERATORS = {
    "ELA (JPEG-90)": lambda x: cv2.cvtColor(ela(x), cv2.COLOR_BGR2RGB),
    "Laplacian": laplacian_residual,
    "Sobel": sobel_edges,
    "DCT log-mag": dct_spectrum,
    "FFT log-mag": fft_spectrum,
}
