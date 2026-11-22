"""Method 0 — Classical DIP colour-transfer baselines, implemented from scratch.

These baselines transfer *global* colour statistics only. They ignore spatial
structure, local contrast and texture, which is exactly why they can fail on
mismatched scenes (e.g. a reference whose sky is huge and a content image that is
mostly foreground): the statistics of the whole image get forced onto the whole
image. That failure motivates the deep, feature-based methods.

Theory recap
------------
**Colour spaces.** RGB channels are strongly correlated (a brightness change moves all
three). CIE L*a*b* separates lightness (L*) from two opponent chroma axes
(a*: green–red, b*: blue–yellow) and is approximately perceptually uniform, so
matching per-channel statistics in Lab rarely creates cross-channel colour casts.
(Reinhard et al. used lαβ, a closely related decorrelated opponent space; CIELab is
the standard modern substitute.)

**Histograms and CDF matching.** The normalised histogram p(r) of a channel is a
discrete PDF; its cumulative sum is the CDF, T(r) = Σ_{k≤r} p(k), which maps
values to [0,1] (this mapping is histogram *equalisation*). To make the content
histogram look like the reference's, apply T_content and then the *inverse* of
the reference CDF: z = G⁻¹(T(r)). The result has (approximately) the reference's
distribution — *histogram specification/matching* (Gonzalez & Woods, §3.3).
"""
from __future__ import annotations

import numpy as np

from .utils import blend, lab_to_rgb, rgb_to_lab

EPS = 1e-6


# --------------------------------------------------------------------------------------
# Reinhard colour transfer
# --------------------------------------------------------------------------------------
def reinhard_transfer(content: np.ndarray, reference: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Reinhard et al. (2001) colour transfer in CIELab.

    For each channel c ∈ {L, a, b}:

        out_c = (x_c − μ_c^content) · (σ_c^ref / σ_c^content) + μ_c^ref

    i.e. a per-channel affine map that matches the first two moments. Note the
    structural similarity to AdaIN — AdaIN does the same thing, but on 512 deep
    feature channels instead of 3 colour channels.

    Args:
        content, reference: float RGB [0,1].
        alpha: strength; 0 returns the content unchanged.
    """
    c_lab = rgb_to_lab(content).reshape(-1, 3).astype(np.float64)
    r_lab = rgb_to_lab(reference).reshape(-1, 3).astype(np.float64)
    mu_c, sd_c = c_lab.mean(0), c_lab.std(0)
    mu_r, sd_r = r_lab.mean(0), r_lab.std(0)
    out = (c_lab - mu_c) * (sd_r / (sd_c + EPS)) + mu_r
    out[:, 0] = np.clip(out[:, 0], 0, 100)
    out_rgb = lab_to_rgb(out.reshape(content.shape).astype(np.float32))
    return blend(content, out_rgb, alpha)


# --------------------------------------------------------------------------------------
# Histogram matching (CDF matching)
# --------------------------------------------------------------------------------------
def match_channel(src: np.ndarray, ref: np.ndarray, bins: int = 1024,
                  lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Match one channel's histogram to a reference channel via CDF inversion.

    Implemented from first principles:
      1. histogram both channels on a common grid of ``bins`` bins,
      2. CDFs by cumulative sum, normalised to [0,1],
      3. for every source value r, find T(r), then z = G⁻¹(T(r)) by linear
         interpolation of the (monotone) reference CDF.

    Using interpolation instead of hard bin look-ups avoids posterisation.
    """
    src_f = src.ravel().astype(np.float64)
    ref_f = ref.ravel().astype(np.float64)
    lo = min(src_f.min(), ref_f.min()) if lo is None else lo
    hi = max(src_f.max(), ref_f.max()) if hi is None else hi
    if hi - lo < EPS:
        return src.copy()
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h_src, _ = np.histogram(src_f, bins=edges)
    h_ref, _ = np.histogram(ref_f, bins=edges)
    cdf_src = np.cumsum(h_src).astype(np.float64)
    cdf_src /= cdf_src[-1]
    cdf_ref = np.cumsum(h_ref).astype(np.float64)
    cdf_ref /= cdf_ref[-1]
    # T(r): CDF of the source at each pixel value (interpolated between bin centres)
    t = np.interp(src_f, centers, cdf_src)
    # G^-1: invert the reference CDF. Keep only strictly increasing points so the
    # inverse function is well defined (flat CDF segments = empty bins).
    keep = np.concatenate([[True], np.diff(cdf_ref) > 0])
    z = np.interp(t, cdf_ref[keep], centers[keep])
    return z.reshape(src.shape).astype(np.float32)


def histogram_matching(content: np.ndarray, reference: np.ndarray, space: str = "rgb",
                       alpha: float = 1.0) -> np.ndarray:
    """Per-channel histogram specification in RGB or Lab.

    * ``rgb``: match R, G, B independently. Simple, but because RGB channels are
      correlated this can introduce hue shifts / colour casts.
    * ``lab``: match L*, a*, b* independently — decorrelated axes, usually more natural.
    """
    if space not in ("rgb", "lab"):
        raise ValueError("space must be 'rgb' or 'lab'")
    if space == "rgb":
        out = np.stack([match_channel(content[..., c], reference[..., c], lo=0.0, hi=1.0)
                        for c in range(3)], axis=-1)
        out = np.clip(out, 0, 1)
    else:
        c_lab, r_lab = rgb_to_lab(content), rgb_to_lab(reference)
        out_lab = np.stack([match_channel(c_lab[..., c], r_lab[..., c]) for c in range(3)], axis=-1)
        out = lab_to_rgb(out_lab)
    return blend(content, out.astype(np.float32), alpha)


def histogram_matching_rgb(content: np.ndarray, reference: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Convenience wrapper: RGB histogram matching."""
    return histogram_matching(content, reference, "rgb", alpha)


def histogram_matching_lab(content: np.ndarray, reference: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Convenience wrapper: Lab histogram matching."""
    return histogram_matching(content, reference, "lab", alpha)
