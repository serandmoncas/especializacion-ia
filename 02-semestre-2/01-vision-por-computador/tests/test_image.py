import numpy as np
import pytest
from applied_cv.image import (
    color_histogram,
    conv2d,
    lbp_histogram,
    local_binary_pattern,
    rgb_to_gray,
    rgb_to_hsv,
    sobel,
)
from scipy import ndimage as ndi

RNG = np.random.default_rng(0)


def test_conv2d_matches_scipy_correlate():
    img, k = RNG.random((13, 17)), RNG.random((3, 5))
    np.testing.assert_allclose(conv2d(img, k), ndi.correlate(img, k, mode="reflect"), atol=1e-12)


def test_sobel_matches_scipy():
    img = RNG.random((10, 12))
    ref = np.hypot(ndi.sobel(img, axis=1, mode="reflect"), ndi.sobel(img, axis=0, mode="reflect"))
    np.testing.assert_allclose(sobel(img), ref, atol=1e-12)


def test_rgb_to_hsv_matches_skimage_including_gray_and_black():
    color = pytest.importorskip("skimage.color")
    rgb = RNG.random((20, 20, 3))
    rgb[0, 0], rgb[0, 1] = 0.5, 0.0  # gris (sin tono) y negro (sin saturación)
    np.testing.assert_allclose(rgb_to_hsv(rgb), color.rgb2hsv(rgb), atol=1e-12)


def test_rgb_to_gray_matches_skimage():
    color = pytest.importorskip("skimage.color")
    rgb = RNG.random((8, 9, 3))
    np.testing.assert_allclose(rgb_to_gray(rgb), color.rgb2gray(rgb), atol=1e-12)


def test_lbp_matches_skimage_bit_for_bit():
    feature = pytest.importorskip("skimage.feature")
    gray = (RNG.random((15, 12)) * 255).round().astype(np.uint8)
    ref = feature.local_binary_pattern(gray, 8, 1, method="default")
    np.testing.assert_array_equal(local_binary_pattern(gray), ref)


def test_histograms_are_normalized():
    rgb = RNG.random((16, 16, 3))
    hist = color_histogram(rgb, bins=8)
    assert hist.shape == (24,)
    np.testing.assert_allclose(hist.reshape(3, 8).sum(axis=1), 1.0)
    lbp = lbp_histogram(np.round(rgb_to_gray(rgb) * 255))
    assert lbp.shape == (256,) and np.isclose(lbp.sum(), 1.0)
