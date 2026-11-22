"""Method 3 — Mood-aware enhancements on top of the deep transfer.

Why "mood" ≠ "style": the mood/atmosphere of a photo (golden hour, blue dusk, fog,
neon night, film, noir) is carried mostly by *global colour, tone and lighting*,
not by brush-stroke texture. Plain AdaIN transfers both and therefore paints
textures onto photos. This module separates the two:

1. **Transfer modes** (applied to the output of *any* method):
   * ``full``       – the raw transfer output (colour + tone + texture).
   * ``color``      – keep the content's luminance (structure), take chroma (a*, b*) from
                      the deep output. Default for "AdaIN + mood".
   * ``tone``       – keep the content's colours (a*, b*), remap its luminance with a
                      global monotone tone curve (CDF matching) towards the deep output's
                      luminance distribution — transfers brightness/contrast only.
   * ``color_tone`` – chroma from the deep output + tone curve on luminance: the most
                      complete *photographic* mood (structure kept by construction, because
                      a global monotone curve never creates or moves edges).
2. **Photorealistic refinement** — a guided filter (He et al., 2010/2013; r = long side/16,
   ε = 1e-2) with the
   *content image as guide*. Inside each window the output is forced to be a local
   affine function of the content colours, q = aᵀI + b, so edges can only appear where
   the content has edges and AdaIN's hallucinated textures are smoothed away. This
   is the smoothing idea of PhotoWCT (Li et al., 2018) and the locally-affine colour
   model of Deep Photo Style Transfer (Luan et al., 2017).
3. **Mood descriptor** — Lab k-means palette + brightness, contrast, saturation,
   colour temperature (McCamy CCT) and a rule-based label.
4. **Presets** — named references in ``data/moods/``.
5. **Post-processing** — histogram refinement, bilateral smoothing, unsharp-mask
   sharpening, film grain, vignette.
6. **Semantic-aware transfer** (stretch) — DeepLabV3 foreground/background mask,
   different mood per region, soft blending.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from .classical import match_channel
from .utils import (DATA_DIR, blend, get_device, lab_to_rgb, load_image, logger, resize_max_side,
                    resize_to, rgb_to_lab, to_uint8)

MODES = ("full", "color", "tone", "color_tone")
MOODS_DIR = DATA_DIR / "moods"


# ======================================================================================
# 1. Guided filter (implemented from the equations of He, Sun & Tang, TPAMI 2013)
# ======================================================================================
def _box(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)×(2r+1) window (O(1) per pixel via OpenCV's running sums)."""
    return cv2.boxFilter(x, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def guided_filter_gray(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter with a single-channel guide I (H,W) and input p (H,W) or (H,W,C).

    Local linear model q_i = a_k I_i + b_k in every window ω_k, with
        a_k = cov(I, p) / (var(I) + ε),   b_k = mean(p) − a_k mean(I).
    """
    if p.ndim == 3:
        return np.stack([guided_filter_gray(I, p[..., c], r, eps) for c in range(p.shape[2])], -1)
    mI, mp = _box(I, r), _box(p, r)
    cov_Ip = _box(I * p, r) - mI * mp
    var_I = _box(I * I, r) - mI * mI
    a = cov_Ip / (var_I + eps)
    b = mp - a * mI
    return (_box(a, r) * I + _box(b, r)).astype(np.float32)


def guided_filter_color(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter with a 3-channel colour guide I (H,W,3) and input p (H,W) or (H,W,C).

    a_k = (Σ_k + εU)⁻¹ cov_k(I, p)   (3-vector),   b_k = mean(p) − a_kᵀ mean(I).
    The 3×3 inverse is computed in closed form (adjugate / determinant) for all
    pixels at once. A colour guide preserves edges between regions of equal
    brightness but different hue, which a grey guide would blur.
    """
    I = I.astype(np.float32)
    if p.ndim == 3:
        return np.stack([guided_filter_color(I, p[..., c], r, eps) for c in range(p.shape[2])], -1)
    p = p.astype(np.float32)
    mI = [_box(I[..., c], r) for c in range(3)]
    mp = _box(p, r)
    cov_Ip = [_box(I[..., c] * p, r) - mI[c] * mp for c in range(3)]
    # covariance entries of the guide
    v = {}
    for i in range(3):
        for j in range(i, 3):
            v[i, j] = _box(I[..., i] * I[..., j], r) - mI[i] * mI[j] + (eps if i == j else 0.0)
    rr, rg, rb, gg, gb, bb = v[0, 0], v[0, 1], v[0, 2], v[1, 1], v[1, 2], v[2, 2]
    # adjugate of the symmetric matrix [[rr,rg,rb],[rg,gg,gb],[rb,gb,bb]]
    i_rr = gg * bb - gb * gb
    i_rg = gb * rb - rg * bb
    i_rb = rg * gb - gg * rb
    i_gg = rr * bb - rb * rb
    i_gb = rb * rg - rr * gb
    i_bb = rr * gg - rg * rg
    det = rr * i_rr + rg * i_rg + rb * i_rb
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    a0 = (i_rr * cov_Ip[0] + i_rg * cov_Ip[1] + i_rb * cov_Ip[2]) / det
    a1 = (i_rg * cov_Ip[0] + i_gg * cov_Ip[1] + i_gb * cov_Ip[2]) / det
    a2 = (i_rb * cov_Ip[0] + i_gb * cov_Ip[1] + i_bb * cov_Ip[2]) / det
    b = mp - a0 * mI[0] - a1 * mI[1] - a2 * mI[2]
    q = _box(a0, r) * I[..., 0] + _box(a1, r) * I[..., 1] + _box(a2, r) * I[..., 2] + _box(b, r)
    return q.astype(np.float32)


def default_radius(img: np.ndarray) -> int:
    """Guided-filter radius scaled with resolution (≈8 px at 512 px long side)."""
    return max(2, int(round(max(img.shape[:2]) / 64)))


def refine_radius(img: np.ndarray) -> int:
    """Guided-filter radius for photorealistic refinement: long side / 16 (32 px at 512 px).

    Chosen empirically (see report §6): small windows (r≈8) keep AdaIN's colour speckle
    in textured regions; r≈long/16 with ε=1e-2 removes speckle and low-frequency colour
    blotches smaller than the window while edges still follow the content.
    """
    return max(2, int(round(max(img.shape[:2]) / 16)))


def photoreal_refine(content: np.ndarray, stylized: np.ndarray, method: str = "guided",
                     r: int | None = None, eps: float = 1e-2) -> np.ndarray:
    """Remove transfer texture and restore content edges (all RGB channels).

    Args:
        method: ``guided`` (colour guided filter, content as guide) or ``bilateral``
            (OpenCV bilateral filter on the stylised image, guide-free; weaker).
    """
    stylized = resize_to(stylized, content.shape[:2])
    if method == "guided":
        r = r or refine_radius(content)
        return np.clip(guided_filter_color(content, stylized, r, eps), 0, 1)
    if method == "bilateral":
        d = 2 * (r or default_radius(content)) + 1
        return np.clip(cv2.bilateralFilter(stylized, d, 0.1, d), 0, 1).astype(np.float32)
    raise ValueError("method must be 'guided' or 'bilateral'")


# ======================================================================================
# 2. Transfer modes
# ======================================================================================
def tone_curve(src_L: np.ndarray, target_L: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Global monotone tone curve mapping src lightness towards target's distribution.

    CDF matching (histogram specification) of L*, softened by ``strength`` and
    lightly regularised: the curve is a 1-D monotone function, so pixel ordering
    (and therefore every edge and texture of the content) is preserved.
    """
    matched = match_channel(src_L, target_L, bins=256, lo=0.0, hi=100.0)
    return np.clip((1 - strength) * src_L + strength * matched, 0, 100).astype(np.float32)


def apply_mode(content: np.ndarray, stylized: np.ndarray, mode: str = "color",
               refine: bool = True, refine_method: str = "guided",
               tone_strength: float = 1.0) -> np.ndarray:
    """Combine the content image with a transfer output according to a transfer mode.

    Photorealistic refinement (if ``refine``) is applied to whatever the mode takes from
    the stylised image (all channels for ``full``, chroma for ``color``/``color_tone``,
    the tone target for ``tone``).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    stylized = resize_to(stylized, content.shape[:2])
    src = photoreal_refine(content, stylized, refine_method) if refine else stylized
    if mode == "full":
        return src
    c_lab, s_lab = rgb_to_lab(content), rgb_to_lab(src)
    out = c_lab.copy()
    if mode in ("color", "color_tone"):
        out[..., 1:] = s_lab[..., 1:]                     # chroma from the deep output
    if mode in ("tone", "color_tone"):
        out[..., 0] = tone_curve(c_lab[..., 0], s_lab[..., 0], tone_strength)
        if mode == "tone":
            # keep perceived saturation roughly constant when lightness changes
            ratio = (out[..., 0] + 5) / (c_lab[..., 0] + 5)
            out[..., 1:] = c_lab[..., 1:] * np.clip(ratio, 0.5, 1.5)[..., None] ** 0.5
    return lab_to_rgb(out)


# ======================================================================================
# 3. Post-processing
# ======================================================================================
def histogram_refine(img: np.ndarray, reference: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Pull the chroma (a*, b*) histograms of ``img`` towards the reference's (CDF matching)."""
    if strength <= 0:
        return img
    lab, ref = rgb_to_lab(img), rgb_to_lab(reference)
    out = lab.copy()
    for c in (1, 2):
        out[..., c] = (1 - strength) * lab[..., c] + strength * match_channel(lab[..., c], ref[..., c])
    return lab_to_rgb(out)


def bilateral_smooth(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Edge-preserving smoothing; ``amount`` ∈ [0,1] blends towards a bilateral-filtered copy."""
    if amount <= 0:
        return img
    d = default_radius(img) + 3
    f = cv2.bilateralFilter(img.astype(np.float32), d, 0.08, d)
    return blend(img, f, amount)


def sharpen(img: np.ndarray, amount: float = 0.5, sigma: float = 1.0) -> np.ndarray:
    """Unsharp masking: out = img + amount · (img − G_σ * img) — boosts high frequencies."""
    if amount <= 0:
        return img
    low = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(img + amount * (img - low), 0, 1).astype(np.float32)


def film_grain(img: np.ndarray, amount: float = 0.3, seed: int = 0) -> np.ndarray:
    """Luminance film grain: slightly blurred Gaussian noise, strongest in mid-tones."""
    if amount <= 0:
        return img
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.standard_normal(img.shape[:2]).astype(np.float32), (0, 0), 0.6)
    y = img.mean(axis=2, keepdims=True)
    mid = 4 * y * (1 - y)                          # parabola: 0 at black/white, 1 at mid-grey
    return np.clip(img + amount * 0.08 * noise[..., None] * (0.3 + 0.7 * mid), 0, 1).astype(np.float32)


def vignette(img: np.ndarray, amount: float = 0.3) -> np.ndarray:
    """Radial darkening towards the corners (cos⁴-like falloff)."""
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    mask = 1 - amount * np.clip(rr, 0, 1) ** 2.2
    return np.clip(img * mask[..., None], 0, 1).astype(np.float32)


@dataclass
class PostConfig:
    """Optional post-processing knobs (all 0 = off)."""
    hist_refine: float = 0.0
    smooth: float = 0.0
    sharpen: float = 0.0
    grain: float = 0.0
    vignette: float = 0.0
    seed: int = 0

    def is_noop(self) -> bool:
        return max(self.hist_refine, self.smooth, self.sharpen, self.grain, self.vignette) <= 0


def post_process(img: np.ndarray, reference: np.ndarray | None, cfg: PostConfig) -> np.ndarray:
    """Apply the enabled post-processing steps in a sensible order."""
    if reference is not None and cfg.hist_refine > 0:
        img = histogram_refine(img, reference, cfg.hist_refine)
    img = bilateral_smooth(img, cfg.smooth)
    img = sharpen(img, cfg.sharpen)
    img = vignette(img, cfg.vignette)
    img = film_grain(img, cfg.grain, cfg.seed)
    return img


# ======================================================================================
# 4. Mood descriptor
# ======================================================================================
@dataclass
class MoodDescriptor:
    """Global mood statistics of an image."""
    palette: np.ndarray            # (k,3) RGB [0,1], sorted by share
    shares: np.ndarray             # (k,) fraction of pixels per palette colour
    brightness: float              # mean L*/100
    contrast: float                # std L*/100 (RMS contrast)
    saturation: float              # mean HSV saturation
    colorfulness: float            # Hasler & Süsstrunk (2003) metric
    cct: float                     # correlated colour temperature (K), McCamy
    warmth: float                  # mean b* + 0.5 mean a*  (>0 warm, <0 cool)
    label: str = ""
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"label": self.label, "brightness": round(self.brightness, 3),
                "contrast": round(self.contrast, 3), "saturation": round(self.saturation, 3),
                "colorfulness": round(self.colorfulness, 1), "cct_K": int(self.cct),
                "warmth": round(self.warmth, 2),
                "palette_hex": ["#%02x%02x%02x" % tuple(to_uint8(c)) for c in self.palette],
                "palette_share": [round(float(s), 3) for s in self.shares]}


def _mccamy_cct(img: np.ndarray) -> float:
    """Correlated colour temperature of the image's mean colour (McCamy 1992).

    sRGB → linear RGB → CIE XYZ → chromaticity (x, y); n = (x − 0.3320)/(0.1858 − y);
    CCT = 449n³ + 3525n² + 6823.3n + 5520.33.  (Gray-world estimate of the illuminant.)
    """
    lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    X, Y, Z = m @ lin.reshape(-1, 3).mean(0)
    s = X + Y + Z + 1e-9
    x, y = X / s, Y / s
    n = (x - 0.3320) / (0.1858 - y + 1e-9)
    return float(np.clip(449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33, 1000, 25000))


def extract_palette(img: np.ndarray, k: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dominant colours via k-means clustering in Lab (perceptual distances)."""
    small = resize_max_side(img, 160)
    lab = rgb_to_lab(small).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(lab, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / labels.size
    order = np.argsort(-shares)
    rgb = lab_to_rgb(centers[order].reshape(1, k, 3)).reshape(k, 3)
    return rgb, shares[order]


def describe_mood(img: np.ndarray, k: int = 5) -> MoodDescriptor:
    """Compute the mood descriptor and a rule-based textual label."""
    img = resize_max_side(img, 384)
    lab = rgb_to_lab(img)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rg = img[..., 0] - img[..., 1]
    yb = 0.5 * (img[..., 0] + img[..., 1]) - img[..., 2]
    colorful = (np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) * 255
    pal, shares = extract_palette(img, k)
    d = MoodDescriptor(palette=pal, shares=shares, brightness=float(L.mean() / 100),
                       contrast=float(L.std() / 100), saturation=float(hsv[..., 1].mean()),
                       colorfulness=float(colorful), cct=_mccamy_cct(img),
                       warmth=float(b.mean() + 0.5 * a.mean()),
                       extras={"a_mean": float(a.mean()), "b_mean": float(b.mean())})
    d.label = mood_label(d)
    return d


def mood_label(d: MoodDescriptor) -> str:
    """Rule-based label from the descriptor (thresholds chosen on the preset set)."""
    parts = []
    a_mean, b_mean = d.extras.get("a_mean", 0.0), d.extras.get("b_mean", 0.0)
    if a_mean > 10 and b_mean < 8:          # pink/purple casts (neon) are not "warm" sunlight
        parts.append("magenta")
    elif b_mean > 8 or d.cct < 4200:        # yellow–orange (b* > 0) = warm light
        parts.append("warm")
    elif b_mean < -4 or d.cct > 7500:       # blue (b* < 0) = cool light
        parts.append("cool")
    else:
        parts.append("neutral")
    parts.append("dark" if d.brightness < 0.35 else "bright" if d.brightness > 0.62 else "mid-key")
    if d.saturation < 0.12:
        parts.append("desaturated")
    elif d.saturation > 0.45 or d.colorfulness > 60:
        parts.append("vivid")
    else:
        parts.append("muted")
    parts.append("high-contrast" if d.contrast > 0.27 else "low-contrast" if d.contrast < 0.14 else "")
    return ", ".join(p for p in parts if p)


def palette_image(d: MoodDescriptor, width: int = 500, height: int = 70) -> np.ndarray:
    """Palette swatch strip with widths proportional to each colour's share."""
    strip = np.zeros((height, width, 3), np.float32)
    x = 0
    for i, (c, s) in enumerate(zip(d.palette, d.shares)):
        w = width - x if i == len(d.palette) - 1 else int(round(s * width))
        strip[:, x:x + w] = c
        x += w
    return strip


def describe_markdown(d: MoodDescriptor) -> str:
    """Human-readable summary used by the GUI."""
    m = d.as_dict()
    return (f"**Mood:** {m['label']}  \n"
            f"Brightness {m['brightness']:.2f} · Contrast {m['contrast']:.2f} · "
            f"Saturation {m['saturation']:.2f} · Colourfulness {m['colorfulness']:.0f} · "
            f"CCT ≈ {m['cct_K']} K · Warmth {m['warmth']:+.1f}  \n"
            f"Palette: {' '.join(m['palette_hex'])}")


# ======================================================================================
# 5. Presets
# ======================================================================================
PRESET_NAMES = {
    "golden_hour": "Golden Hour", "blue_dusk": "Cold Blue Dusk", "foggy": "Foggy / Misty",
    "neon_night": "Neon Night", "vintage_film": "Vintage Film", "dark_noir": "Dark Noir",
}


def list_presets(folder: Path = MOODS_DIR) -> dict[str, Path]:
    """Display name → path for every mood reference in ``data/moods``."""
    out = {}
    if folder.exists():
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                out[PRESET_NAMES.get(p.stem, p.stem.replace("_", " ").title())] = p
    return out


def load_preset(name: str, max_side: int | None = 1024) -> np.ndarray:
    """Load a preset by display name or file stem."""
    presets = list_presets()
    for disp, p in presets.items():
        if name in (disp, p.stem):
            return load_image(p, max_side)
    raise KeyError(f"Unknown mood preset {name!r}; available: {list(presets)}")


# ======================================================================================
# 6. Full "AdaIN + mood enhancements" pipeline
# ======================================================================================
@dataclass
class MoodConfig:
    """Settings of the AdaIN + mood pipeline."""
    alpha: float = 1.0
    mode: str = "color"
    refine: bool = True
    refine_method: str = "guided"
    tone_strength: float = 1.0
    size: int | None = 512
    weights: Sequence[float] | None = None
    post: PostConfig = field(default_factory=PostConfig)


def adain_mood_transfer(content: np.ndarray, refs: np.ndarray | Sequence[np.ndarray],
                        cfg: MoodConfig | None = None, device=None) -> np.ndarray:
    """AdaIN (deep) + transfer mode + photorealistic refinement + post-processing."""
    from .adain import get_adain
    cfg = cfg or MoodConfig()
    if isinstance(refs, np.ndarray):
        refs = [refs]
    if cfg.size is not None:
        content = resize_max_side(content, cfg.size)
    model = get_adain(device)
    stylized = model.stylize(content, refs, alpha=cfg.alpha, size=None if cfg.size is None else cfg.size,
                             weights=cfg.weights)
    out = apply_mode(content, stylized, cfg.mode, cfg.refine, cfg.refine_method, cfg.tone_strength)
    if not cfg.post.is_noop():
        out = post_process(out, refs[0], cfg.post)
    return out


# ======================================================================================
# 7. Semantic-aware transfer (stretch goal)
# ======================================================================================
_SEG_CACHE: dict[str, torch.nn.Module] = {}


def semantic_mask(img: np.ndarray, device=None) -> np.ndarray:
    """Soft foreground probability (H,W) in [0,1] from DeepLabV3-MobileNetV3.

    The torchvision model predicts the 20 PASCAL-VOC object classes + background.
    VOC has **no 'sky' class**, so we split the image into *foreground objects*
    (people, animals, vehicles, furniture, …) vs *background* (sky, buildings,
    vegetation, ground). The probability map is refined with a guided filter
    (content as guide) so the mask snaps to object edges, then used as a soft blend.
    """
    import torchvision
    from .models import TORCH_HUB_DIR
    dev = torch.device(device) if device is not None else get_device()
    key = str(dev)
    if key not in _SEG_CACHE:
        torch.hub.set_dir(str(TORCH_HUB_DIR))
        w = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        _SEG_CACHE[key] = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=w).eval().to(dev)
    net = _SEG_CACHE[key]
    small = resize_max_side(img, 520)
    x = torch.from_numpy(small.transpose(2, 0, 1)).unsqueeze(0).to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    with torch.no_grad():
        prob = net((x - mean) / std)["out"].softmax(1)[0]
    fg = (1 - prob[0]).cpu().numpy().astype(np.float32)          # 1 − P(background)
    fg = resize_to(np.repeat(fg[..., None], 3, 2), img.shape[:2])[..., 0]
    fg = guided_filter_gray(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), fg, default_radius(img) * 2, 1e-3)
    return np.clip(fg, 0, 1)


def semantic_mood_transfer(content: np.ndarray, ref_background: np.ndarray,
                           ref_foreground: np.ndarray | None = None, fg_alpha: float = 0.35,
                           cfg: MoodConfig | None = None, device=None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Apply a mood to the background and a (weaker or different) mood to foreground objects.

    Returns (output, foreground mask).
    """
    cfg = cfg or MoodConfig()
    if cfg.size is not None:
        content = resize_max_side(content, cfg.size)
    bg = adain_mood_transfer(content, ref_background, cfg, device)
    fg_cfg = MoodConfig(**{**cfg.__dict__, "alpha": fg_alpha if ref_foreground is None else cfg.alpha})
    fg = adain_mood_transfer(content, ref_foreground if ref_foreground is not None else ref_background,
                             fg_cfg, device)
    m = semantic_mask(content, device)[..., None]
    return np.clip(m * fg + (1 - m) * bg, 0, 1).astype(np.float32), m[..., 0]


def timed(fn, *args, **kwargs):
    """Run ``fn`` and return (result, seconds)."""
    t0 = time.perf_counter()
    r = fn(*args, **kwargs)
    return r, time.perf_counter() - t0
