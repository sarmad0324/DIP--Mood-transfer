"""Network definitions: VGG-19 encoders and the AdaIN decoder.

Two *different* VGG-19 weight sets are used in this project, on purpose:

1. **AdaIN (Method 2)** uses the *normalised* VGG-19 (``vgg_normalised.pth``) released
   with the pretrained AdaIN decoder (Huang & Belongie 2017; PyTorch port by
   naoto0804/pytorch-AdaIN). The decoder was trained to invert *exactly* these
   features, so the encoder must match it. Its first layer is a 1×1 convolution that
   already contains the original Caffe preprocessing (RGB→BGR swap, ×255, mean
   subtraction), so the input is **plain RGB in [0, 1] with no ImageNet
   normalisation**. Feeding torchvision-VGG features to this decoder produces garbage.

2. **Gatys (Method 1)** uses **torchvision's** ImageNet VGG-19
   (``VGG19_Weights.IMAGENET1K_V1``), whose input must be normalised with the
   ImageNet mean/std. Gatys only *optimises pixels*, so any good VGG works.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .utils import WEIGHTS_DIR

ADAIN_DECODER_PATH = WEIGHTS_DIR / "adain" / "decoder.pth"
ADAIN_VGG_PATH = WEIGHTS_DIR / "adain" / "vgg_normalised.pth"
TORCH_HUB_DIR = WEIGHTS_DIR / "torch_hub"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------------------
# AdaIN encoder / decoder (architecture must match the released state dicts exactly)
# --------------------------------------------------------------------------------------
def build_adain_vgg() -> nn.Sequential:
    """Full normalised VGG-19 (53 modules, indices must match ``vgg_normalised.pth``).

    Reflection padding (instead of zero padding) avoids border artefacts.
    ``ceil_mode=True`` pooling matches the original Torch7 model.
    """
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),                       # 0: built-in preprocessing
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),       # relu1_1 (3)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),      # relu1_2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),     # relu2_1 (10)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),    # relu2_2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),    # relu3_1 (17)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),    # relu3_2
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),    # relu3_3
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),    # relu3_4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),    # relu4_1 (30)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),    # relu4_2
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),    # relu4_3
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),    # relu4_4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),    # relu5_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),    # relu5_2
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),    # relu5_3
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),    # relu5_4
    )


def build_adain_decoder() -> nn.Sequential:
    """Decoder that mirrors the encoder up to relu4_1 (nearest upsampling ×3 = ×8).

    Nearest-neighbour upsampling + conv avoids the checkerboard artefacts of
    transposed convolutions; reflection padding avoids border artefacts.
    """
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
    )


class AdaINEncoder(nn.Module):
    """Frozen normalised VGG-19 truncated at relu4_1 (input: RGB in [0,1])."""

    #: slice boundaries of relu1_1, relu2_1, relu3_1, relu4_1 inside the Sequential
    SLICES = ((0, 4), (4, 11), (11, 18), (18, 31))

    def __init__(self, weights: Path | None = ADAIN_VGG_PATH) -> None:
        super().__init__()
        vgg = build_adain_vgg()
        if weights is not None:
            vgg.load_state_dict(_load_state(weights))
        layers = list(vgg.children())
        self.stages = nn.ModuleList(nn.Sequential(*layers[a:b]) for a, b in self.SLICES)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor, all_layers: bool = False):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats if all_layers else feats[-1]


class AdaINDecoder(nn.Module):
    """Pretrained AdaIN decoder (relu4_1 features -> RGB image)."""

    def __init__(self, weights: Path | None = ADAIN_DECODER_PATH) -> None:
        super().__init__()
        self.net = build_adain_decoder()
        if weights is not None:
            self.net.load_state_dict(_load_state(weights))
        self.eval()

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


def _load_state(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing weights: {path}\nRun:  python scripts/download_weights.py")
    return torch.load(path, map_location="cpu", weights_only=True)


# --------------------------------------------------------------------------------------
# torchvision VGG-19 feature extractor (Gatys)
# --------------------------------------------------------------------------------------
#: torchvision ``vgg19().features`` indices of the ReLU outputs we use
TV_VGG_LAYERS = {
    "relu1_1": 1, "relu2_1": 6, "relu3_1": 11, "relu4_1": 20,
    "relu4_2": 22, "relu5_1": 29,
}


class TorchvisionVGG19(nn.Module):
    """torchvision ImageNet VGG-19 returning a dict of named ReLU activations.

    Input: RGB in [0,1]; ImageNet mean/std normalisation is applied inside.
    Max-pooling is optionally replaced by average pooling, as suggested by
    Gatys et al. (2016) for smoother gradients.
    """

    def __init__(self, layers: tuple[str, ...], avg_pool: bool = True) -> None:
        super().__init__()
        import torchvision
        torch.hub.set_dir(str(TORCH_HUB_DIR))
        try:
            vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        except Exception as e:  # pragma: no cover - network failure path
            raise RuntimeError("Could not load torchvision VGG-19 weights. "
                               "Run `python scripts/download_weights.py` with internet.") from e
        feats = vgg.features
        last = max(TV_VGG_LAYERS[l] for l in layers)
        mods = []
        for i, m in enumerate(feats[: last + 1]):
            if avg_pool and isinstance(m, nn.MaxPool2d):
                m = nn.AvgPool2d(kernel_size=2, stride=2)
            if isinstance(m, nn.ReLU):
                m = nn.ReLU(inplace=False)
            mods.append(m)
        self.body = nn.Sequential(*mods)
        self.wanted = {TV_VGG_LAYERS[l]: l for l in layers}
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x - self.mean) / self.std
        out = {}
        for i, m in enumerate(self.body):
            x = m(x)
            if i in self.wanted:
                out[self.wanted[i]] = x
        return out
