import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepfake_detection.forensics import (  # noqa: E402
    OPERATORS,
    dct_spectrum,
    ela,
    ela_energy,
    fft_spectrum,
    laplacian_residual,
    sobel_edges,
)


@pytest.fixture
def textured():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def flat():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


def test_ela_shape_dtype_and_flat_image_is_near_zero(textured, flat):
    out = ela(textured)
    assert out.shape == textured.shape and out.dtype == np.uint8
    assert ela_energy(flat) < 1.0
    assert ela_energy(textured) > ela_energy(flat)


def test_ela_energy_drops_after_heavy_recompression(textured):
    ok, enc = cv2.imencode(".jpg", textured, [cv2.IMWRITE_JPEG_QUALITY, 90])
    once = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    assert ela_energy(once) < ela_energy(textured)


def test_spectra_are_finite_and_2d(textured):
    for fn in (dct_spectrum, fft_spectrum):
        out = fn(textured)
        assert out.ndim == 2 and np.isfinite(out).all()


def test_dct_handles_odd_sizes():
    img = np.zeros((63, 65, 3), dtype=np.uint8)
    assert dct_spectrum(img).shape == (62, 64)


def test_high_pass_filters_are_zero_on_flat_images(flat, textured):
    assert laplacian_residual(flat).max() == 0
    assert sobel_edges(flat).max() == 0
    assert sobel_edges(textured).mean() > 0


def test_operator_registry_runs_every_operator(textured):
    assert len(OPERATORS) == 5
    for name, op in OPERATORS.items():
        assert op(textured).size > 0, name
