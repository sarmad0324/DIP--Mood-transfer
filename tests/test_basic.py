"""Smoke tests for every method (small 128 px images so the suite runs in seconds).

Run:  pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import classical, mood  # noqa: E402
from src.metrics import bhattacharyya_rgb, lab_stats_distance, ssim_luma, wasserstein_lab  # noqa: E402
from src.utils import (load_image, pad_to_multiple, resize_max_side, rgb_to_lab, lab_to_rgb,  # noqa: E402
                       rgb_to_ycbcr, save_image, set_seed, to_float_rgb, ycbcr_to_rgb)

SIZE = 128
HAS_ADAIN = (ROOT / "weights" / "adain" / "decoder.pth").exists()


@pytest.fixture(scope="module")
def imgs():
    """A natural content photo and a warm/orange reference from skimage (public domain / CC0)."""
    import skimage.data as d
    content = resize_max_side(to_float_rgb(d.astronaut()), SIZE)
    ref = resize_max_side(to_float_rgb(d.coffee()), SIZE)
    return content, ref


def _check(out: np.ndarray, shape=None):
    assert out.dtype == np.float32
    assert out.ndim == 3 and out.shape[2] == 3
    if shape is not None:
        assert out.shape[:2] == shape[:2]
    assert np.isfinite(out).all(), "NaN/Inf in output"
    assert out.min() >= 0.0 and out.max() <= 1.0


# ----------------------------------------------------------------------------- utils
def test_color_space_roundtrip(imgs):
    c, _ = imgs
    assert np.abs(lab_to_rgb(rgb_to_lab(c)) - c).max() < 3e-3  # float32 Lab round trip
    assert np.abs(ycbcr_to_rgb(rgb_to_ycbcr(c)) - c).max() < 2e-3


def test_io_roundtrip(tmp_path, imgs):
    c, _ = imgs
    p = save_image(c, tmp_path / "x.png")
    back = load_image(p)
    assert back.shape == c.shape and np.abs(back - c).max() <= 1 / 255 + 1e-6


def test_load_gray_and_rgba():
    _check(to_float_rgb(np.zeros((10, 12), np.uint8)), (10, 12))
    _check(to_float_rgb(np.zeros((10, 12, 4), np.uint8)), (10, 12))


def test_pad_to_multiple():
    import torch
    t, hw = pad_to_multiple(torch.zeros(1, 3, 37, 50), 8)
    assert t.shape[-2:] == (40, 56) and hw == (37, 50)


# ----------------------------------------------------------------------------- classical
@pytest.mark.parametrize("fn", [classical.reinhard_transfer, classical.histogram_matching_rgb,
                                classical.histogram_matching_lab])
def test_classical_shapes(imgs, fn):
    c, r = imgs
    _check(fn(c, r), c.shape)


@pytest.mark.parametrize("fn", [classical.reinhard_transfer, classical.histogram_matching_rgb,
                                classical.histogram_matching_lab])
def test_classical_alpha0_is_identity(imgs, fn):
    c, r = imgs
    assert np.abs(fn(c, r, alpha=0.0) - c).max() < 1e-5


def test_reinhard_matches_lab_moments(imgs):
    c, r = imgs
    out = classical.reinhard_transfer(c, r)
    d = lab_stats_distance(out, r)
    d0 = lab_stats_distance(c, r)
    assert d["lab_stat_dist"] < 0.3 * d0["lab_stat_dist"]


def test_histogram_matching_distribution(imgs):
    c, r = imgs
    out = classical.histogram_matching_rgb(c, r)
    assert wasserstein_lab(out, r) < wasserstein_lab(c, r)


# ----------------------------------------------------------------------------- mood module
def test_guided_filter_identity_for_constant():
    I = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    p = np.full((32, 32), 0.3, np.float32)
    assert np.allclose(mood.guided_filter_color(I, p, 3, 1e-3), 0.3, atol=1e-4)
    assert np.allclose(mood.guided_filter_gray(I[..., 0], p, 3, 1e-3), 0.3, atol=1e-4)


@pytest.mark.parametrize("mode", mood.MODES)
def test_apply_modes(imgs, mode):
    c, r = imgs
    stylized = classical.reinhard_transfer(c, r)
    _check(mood.apply_mode(c, stylized, mode, refine=True), c.shape)


def test_color_mode_keeps_luminance(imgs):
    c, r = imgs
    out = mood.apply_mode(c, classical.reinhard_transfer(c, r), "color", refine=False)
    assert np.abs(rgb_to_lab(out)[..., 0] - rgb_to_lab(c)[..., 0]).mean() < 2.0  # L* units (gamut clip)


def test_mode_alpha0_is_identity(imgs):
    """Mood-blending path with a zero-strength transfer returns ~ the content."""
    c, r = imgs
    stylized = classical.reinhard_transfer(c, r, alpha=0.0)
    for mode in ("color", "tone", "color_tone"):
        out = mood.apply_mode(c, stylized, mode, refine=False)
        assert np.abs(out - c).mean() < 0.01, mode


def test_post_processing(imgs):
    c, r = imgs
    cfg = mood.PostConfig(hist_refine=0.5, smooth=0.5, sharpen=0.5, grain=0.5, vignette=0.5)
    _check(mood.post_process(c, r, cfg), c.shape)
    assert np.array_equal(mood.post_process(c, r, mood.PostConfig()), c)


def test_descriptor(imgs):
    _, r = imgs
    d = mood.describe_mood(r)
    assert d.palette.shape == (5, 3) and abs(d.shares.sum() - 1) < 1e-6
    assert 0 <= d.brightness <= 1 and 1000 <= d.cct <= 25000
    assert "warm" in d.label  # coffee image is clearly warm
    _check(mood.palette_image(d))


# ----------------------------------------------------------------------------- metrics
def test_metrics_identity(imgs):
    c, _ = imgs
    assert ssim_luma(c, c) > 0.999
    assert bhattacharyya_rgb(c, c) < 1e-6


# ----------------------------------------------------------------------------- deep methods
@pytest.mark.skipif(not HAS_ADAIN, reason="run scripts/download_weights.py first")
def test_adain_shapes_and_blending(imgs):
    from src.adain import get_adain
    c, r = imgs
    m = get_adain("cpu")
    out = m.stylize(c, r, alpha=1.0, size=SIZE)
    _check(out, c.shape)
    out2 = m.stylize(c, [r, np.flipud(r).copy()], alpha=1.0, size=SIZE, weights=[0.7, 0.3])
    _check(out2, c.shape)


@pytest.mark.skipif(not HAS_ADAIN, reason="run scripts/download_weights.py first")
def test_adain_alpha0_reconstruction(imgs):
    """alpha=0 is decoder(encoder(content)) — only an approximate reconstruction.

    The pretrained decoder was trained with content+style losses, never for exact
    reconstruction, and relu4_1 works at 1/8 resolution, so reconstruction SSIM depends
    on size and image detail (measured: 0.29–0.71 at 128–256 px, see README). We use
    the NASA rocket image (public domain) at 192 px, where SSIM ≈ 0.69.
    """
    import skimage.data as d
    from src.adain import get_adain
    _, r = imgs
    c = resize_max_side(to_float_rgb(d.rocket()), 192)
    m = get_adain("cpu")
    a0 = m.stylize(c, r, alpha=0.0, size=192)
    a1 = m.stylize(c, r, alpha=1.0, size=192)
    s0, s1 = ssim_luma(c, a0), ssim_luma(c, a1)
    assert s0 > 0.6, s0
    assert s0 > s1
    assert np.abs(a0 - c).mean() < np.abs(a1 - c).mean()


@pytest.mark.skipif(not HAS_ADAIN, reason="run scripts/download_weights.py first")
def test_adain_tiled_matches_shape():
    from src.adain import get_adain
    set_seed(0)
    img = np.random.default_rng(0).random((200, 150, 3)).astype(np.float32)
    out = get_adain("cpu").stylize(img, img, size=None, tile=96)
    _check(out, img.shape)


@pytest.mark.skipif(not HAS_ADAIN, reason="run scripts/download_weights.py first")
def test_adain_mood_pipeline(imgs):
    c, r = imgs
    from src.pipeline import TransferOptions, transfer
    out, _ = transfer(c, r, TransferOptions(method="adain_mood", size=SIZE), "cpu")
    _check(out, c.shape)
    out0, _ = transfer(c, r, TransferOptions(method="adain_mood", size=SIZE, alpha=0.0), "cpu")
    assert ssim_luma(c, out0) > 0.9 and np.abs(out0 - c).mean() < 0.05


def test_gatys_smoke(imgs):
    pytest.importorskip("torchvision")
    if not (ROOT / "weights" / "torch_hub" / "checkpoints").exists():
        pytest.skip("VGG-19 weights not downloaded")
    from src.gatys import GatysConfig, get_gatys
    c, r = imgs
    out, hist = get_gatys("cpu").stylize(c, r, GatysConfig(iterations=5, size=64), progress=False)
    _check(out)
    assert len(hist["total"]) >= 5 and np.isfinite(hist["total"]).all()


def test_pipeline_all_methods_fast(imgs):
    from src.pipeline import TransferOptions, transfer
    c, r = imgs
    for method in ("reinhard", "hist"):
        out, _ = transfer(c, r, TransferOptions(method=method, size=SIZE))
        _check(out, c.shape)
    with pytest.raises(ValueError):
        transfer(c, r, TransferOptions(method="nope"))
    with pytest.raises(ValueError):
        transfer(c, r, TransferOptions(method="reinhard", alpha=1.5))
