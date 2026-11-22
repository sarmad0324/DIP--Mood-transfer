"""Single entry point that dispatches to every method (used by the CLI, GUI and experiments)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .classical import histogram_matching, reinhard_transfer
from .gatys import GatysConfig, get_gatys
from .mood import MODES, MoodConfig, PostConfig, adain_mood_transfer, apply_mode, post_process
from .utils import blend, resize_max_side

METHODS = ("reinhard", "hist", "gatys", "adain", "adain_mood")
METHOD_LABELS = {"reinhard": "Reinhard", "hist": "Histogram", "gatys": "Gatys",
                 "adain": "AdaIN", "adain_mood": "AdaIN + mood enhancements"}


@dataclass
class TransferOptions:
    """All user-facing options.

    ``mode=None`` means the method's natural default: ``full`` for every method except
    ``adain_mood`` which defaults to ``color`` (+ photorealistic refinement).
    """
    method: str = "adain_mood"
    alpha: float = 1.0
    mode: str | None = None
    refine: bool | None = None
    refine_method: str = "guided"
    size: int = 512
    weights: Sequence[float] | None = None
    hist_space: str = "lab"
    gatys_iters: int = 300
    gatys_style_weight: float = 1e5
    post: PostConfig = field(default_factory=PostConfig)


def transfer(content: np.ndarray, refs: np.ndarray | Sequence[np.ndarray],
             opt: TransferOptions | None = None, device=None) -> tuple[np.ndarray, dict]:
    """Run one method. Returns (output float RGB [0,1], info dict)."""
    opt = opt or TransferOptions()
    if opt.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not 0 <= opt.alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if isinstance(refs, np.ndarray):
        refs = [refs]
    refs = [resize_max_side(r, opt.size) for r in refs]
    content = resize_max_side(content, opt.size)
    info: dict = {}
    mode = opt.mode or ("color" if opt.method == "adain_mood" else "full")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    refine = opt.refine if opt.refine is not None else (opt.method == "adain_mood")

    if opt.method == "adain_mood":
        cfg = MoodConfig(alpha=opt.alpha, mode=mode, refine=refine, refine_method=opt.refine_method,
                         size=opt.size, weights=opt.weights, post=opt.post)
        return adain_mood_transfer(content, refs, cfg, device), info

    if opt.method == "adain":
        from .adain import get_adain
        out = get_adain(device).stylize(content, refs, alpha=opt.alpha, size=opt.size, weights=opt.weights)
    else:
        ref = _blend_refs_pixels(refs, opt.weights) if len(refs) > 1 else refs[0]
        if opt.method == "reinhard":
            out = reinhard_transfer(content, ref, opt.alpha)
        elif opt.method == "hist":
            out = histogram_matching(content, ref, opt.hist_space, opt.alpha)
        else:  # gatys
            cfg = GatysConfig(iterations=opt.gatys_iters, size=opt.size, style_weight=opt.gatys_style_weight)
            out, hist = get_gatys(device).stylize(content, ref, cfg)
            out = blend(resize_max_side(content, opt.size), out, opt.alpha)
            info["loss_history"] = hist
    if mode != "full" or refine:
        out = apply_mode(resize_max_side(content, opt.size), out, mode, refine, opt.refine_method)
    if not opt.post.is_noop():
        out = post_process(out, refs[0], opt.post)
    return out, info


def _blend_refs_pixels(refs: Sequence[np.ndarray], weights: Sequence[float] | None) -> np.ndarray:
    """For non-AdaIN methods, multi-reference = concatenate the references side by side
    weighted by repetition, so their pooled statistics are a weighted mixture."""
    from .utils import resize_to
    w = np.asarray(weights if weights is not None else [1.0] * len(refs), float)
    w = w / w.sum()
    h = 256
    parts = []
    for r, wi in zip(refs, w):
        width = max(1, int(round(wi * 1024)))
        parts.append(resize_to(r, (h, width)))
    return np.concatenate(parts, axis=1)
