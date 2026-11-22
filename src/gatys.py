"""Method 1 — Optimisation-based neural transfer (Gatys, Ecker & Bethge, CVPR 2016).

We optimise the *pixels* of an image x so that

    L(x) = w_c · L_content(x, c) + w_s · L_style(x, s) + w_tv · L_TV(x)

* **Content loss** — feature reconstruction at ``relu4_2``:
      L_content = mean( (F_l(x) − F_l(c))² )
* **Style loss** — Gram matrices G_l = F_l F_lᵀ / (C_l H_l W_l) at
  ``relu1_1 … relu5_1``. The Gram matrix holds the correlations between feature
  channels, averaged over all positions, i.e. *what* features co-occur regardless
  of *where* — a spatially invariant summary of colour/texture/"mood":
      L_style = Σ_l  w_l · mean( (G_l(x) − G_l(s))² )
* **Total variation** — L_TV = mean|∂x/∂u| + mean|∂x/∂v| suppresses high-frequency
  noise (a smoothness prior).

The image is initialised from the **content image**, not noise, so the solution
stays near the photo (more photographic, fewer iterations needed).

Network: torchvision ImageNet VGG-19 with ImageNet normalisation (see models.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .models import TorchvisionVGG19
from .utils import get_device, logger, np_to_tensor, resize_max_side, tensor_to_np, free_memory

STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
CONTENT_LAYER = "relu4_2"


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Normalised Gram matrix of an (N,C,H,W) feature map -> (N,C,C)."""
    n, c, h, w = feat.shape
    f = feat.reshape(n, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation (mean absolute finite differences)."""
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


@dataclass
class GatysConfig:
    """Hyper-parameters of the Gatys optimisation."""
    content_weight: float = 1.0
    style_weight: float = 1e5
    tv_weight: float = 1e-2
    iterations: int = 300        # number of L-BFGS function evaluations
    optimizer: str = "lbfgs"     # "lbfgs" or "adam"
    lr: float = 0.02             # Adam only
    size: int = 384
    log_every: int = 50
    history: list = field(default_factory=list)


class GatysTransfer:
    """Gatys et al. neural transfer with L-BFGS / Adam."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else get_device()
        self.vgg = TorchvisionVGG19(STYLE_LAYERS + (CONTENT_LAYER,)).to(self.device)

    def stylize(self, content: np.ndarray, reference: np.ndarray, cfg: GatysConfig | None = None,
                progress: bool = True) -> tuple[np.ndarray, dict[str, list[float]]]:
        """Run the optimisation.

        Returns:
            (output image float RGB [0,1], loss history dict with keys
             'total', 'content', 'style', 'tv').
        """
        cfg = cfg or GatysConfig()
        content = resize_max_side(content, cfg.size)
        reference = resize_max_side(reference, cfg.size)
        c = np_to_tensor(content, self.device)
        s = np_to_tensor(reference, self.device)
        with torch.no_grad():
            c_feats = self.vgg(c)
            target_content = c_feats[CONTENT_LAYER].detach()
            s_feats = self.vgg(s)
            target_grams = {l: gram_matrix(s_feats[l]).detach() for l in STYLE_LAYERS}

        x = c.clone().requires_grad_(True)  # init from content -> photographic result
        hist: dict[str, list[float]] = {"total": [], "content": [], "style": [], "tv": []}
        layer_w = 1.0 / len(STYLE_LAYERS)

        def compute() -> torch.Tensor:
            feats = self.vgg(x)
            lc = F.mse_loss(feats[CONTENT_LAYER], target_content)
            ls = sum(layer_w * F.mse_loss(gram_matrix(feats[l]), target_grams[l]) for l in STYLE_LAYERS)
            lt = tv_loss(x)
            total = cfg.content_weight * lc + cfg.style_weight * ls + cfg.tv_weight * lt
            hist["total"].append(total.item()); hist["content"].append(lc.item())
            hist["style"].append(ls.item()); hist["tv"].append(lt.item())
            return total

        bar = tqdm(total=cfg.iterations, desc="Gatys", disable=not progress, leave=False)
        t0 = time.time()
        if cfg.optimizer == "lbfgs":
            opt = torch.optim.LBFGS([x], lr=1.0, max_iter=cfg.iterations, history_size=50,
                                    tolerance_grad=-1, tolerance_change=-1)  # run all iterations

            def closure() -> torch.Tensor:
                opt.zero_grad()
                loss = compute()
                loss.backward()
                bar.update(1)
                return loss
            opt.step(closure)
        elif cfg.optimizer == "adam":
            opt = torch.optim.Adam([x], lr=cfg.lr)
            for _ in range(cfg.iterations):
                opt.zero_grad()
                loss = compute()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(0, 1)
                bar.update(1)
        else:
            raise ValueError("optimizer must be 'lbfgs' or 'adam'")
        bar.close()
        logger.debug("Gatys finished %d iters in %.1fs", cfg.iterations, time.time() - t0)
        out = tensor_to_np(x)
        free_memory(self.device)
        return out, hist


def plot_loss_curves(hist: dict[str, list[float]], path: Path | str, title: str = "Gatys losses") -> Path:
    """Save a log-scale plot of the weighted-loss components."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k in ("total", "content", "style", "tv"):
        ax.plot(hist[k], label=k)
    ax.set_yscale("log"); ax.set_xlabel("L-BFGS evaluation"); ax.set_ylabel("loss (unweighted parts)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


_CACHE: dict[str, GatysTransfer] = {}


def get_gatys(device: torch.device | str | None = None) -> GatysTransfer:
    """Cached model per device."""
    dev = str(torch.device(device) if device is not None else get_device())
    if dev not in _CACHE:
        _CACHE[dev] = GatysTransfer(dev)
    return _CACHE[dev]
