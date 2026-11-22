#!/usr/bin/env python
"""Mood Transfer command-line interface.

Single image:
    python run.py --content a.jpg --mood b.jpg --method adain --alpha 0.8 --mode full --out out.png
    python run.py --content a.jpg --preset "Golden Hour" --method adain_mood --out out.png

Multi-reference blending (weights optional):
    python run.py --content a.jpg --mood m1.jpg m2.jpg --weights 0.7 0.3 --out out.png

Batch (folder of contents × set of moods):
    python run.py --batch --content data/content --mood data/moods --method adain_mood --out results/batch

``--style`` is accepted as an alias for ``--mood``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.mood import MODES, PostConfig, describe_mood, list_presets, load_preset
from src.pipeline import METHODS, TransferOptions, transfer
from src.utils import (default_size, get_device, list_images, load_image, save_image, set_seed,
                       setup_logging, sync)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transfer the mood of one image to another.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--content", required=True, help="content image (or folder with --batch)")
    p.add_argument("--mood", "--style", dest="mood", nargs="+",
                   help="mood/reference image(s) (or folder(s) with --batch)")
    p.add_argument("--preset", nargs="+", help="mood preset name(s) from data/moods (instead of --mood)")
    p.add_argument("--weights", type=float, nargs="+", help="blend weights for multiple references")
    p.add_argument("--method", default="adain_mood", choices=METHODS)
    p.add_argument("--alpha", type=float, default=1.0, help="strength in [0,1]")
    p.add_argument("--mode", choices=MODES, default=None,
                   help="transfer mode (default: color for adain_mood, full otherwise)")
    p.add_argument("--refine", dest="refine", action="store_true", default=None,
                   help="force photorealistic refinement on (default on for adain_mood)")
    p.add_argument("--no-refine", dest="refine", action="store_false")
    p.add_argument("--refine-method", choices=("guided", "bilateral"), default="guided")
    p.add_argument("--size", type=int, default=None, help="long side in px (default 512 GPU / 384 CPU)")
    p.add_argument("--iters", type=int, default=300, help="Gatys iterations")
    p.add_argument("--hist-space", choices=("rgb", "lab"), default="lab")
    p.add_argument("--hist-refine", type=float, default=0.0)
    p.add_argument("--smooth", type=float, default=0.0)
    p.add_argument("--sharpen", type=float, default=0.0)
    p.add_argument("--grain", type=float, default=0.0)
    p.add_argument("--vignette", type=float, default=0.0)
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", action="store_true", help="content/mood are folders: run all pairs")
    p.add_argument("--describe", action="store_true", help="also print the mood descriptor of the reference")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--out", default="out.png", help="output file (or folder with --batch)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    log = setup_logging()
    if a.list_presets:
        for name, path in list_presets().items():
            print(f"{name:20s} {path}")
        return 0
    set_seed(a.seed)
    device = get_device(a.device)
    size = a.size or default_size(device)
    log.info("Device: %s | working size: %d px | method: %s", device, size, a.method)
    if not 0 <= a.alpha <= 1:
        log.error("--alpha must be in [0, 1]")
        return 2
    opt = TransferOptions(method=a.method, alpha=a.alpha, mode=a.mode, refine=a.refine,
                          refine_method=a.refine_method, size=size, weights=a.weights,
                          hist_space=a.hist_space, gatys_iters=a.iters,
                          post=PostConfig(a.hist_refine, a.smooth, a.sharpen, a.grain, a.vignette, a.seed))
    if a.method == "gatys" and device.type == "cpu":
        log.warning("Gatys on CPU is slow (~minutes per image). Consider --size 256 --iters 150.")

    # ---------------------------------------------------------------- batch mode
    if a.batch:
        contents = list_images(a.content)
        mood_paths = []
        for m in (a.mood or []):
            mp = Path(m)
            mood_paths += list_images(mp) if mp.is_dir() else [mp]
        if a.preset:
            mood_paths += [list_presets()[n] for n in a.preset]
        if not contents or not mood_paths:
            log.error("Batch mode needs a folder of content images and at least one mood.")
            return 2
        out_dir = Path(a.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        moods = {p.stem: load_image(p, max_side=1024) for p in mood_paths}
        timings = []
        for cp in tqdm(contents, desc="contents"):
            c = load_image(cp, max_side=2048)
            for mname, m in moods.items():
                t0 = time.perf_counter()
                out, _ = transfer(c, m, opt, device)
                sync(device)
                timings.append(time.perf_counter() - t0)
                save_image(out, out_dir / f"{cp.stem}__{mname}__{a.method}.png")
        log.info("Saved %d images to %s (mean %.2fs/img)", len(timings), out_dir, sum(timings) / len(timings))
        return 0

    # ---------------------------------------------------------------- single image
    if not a.mood and not a.preset:
        log.error("Give --mood/--style image(s) or --preset name(s).")
        return 2
    try:
        content = load_image(a.content, max_side=2048)
        refs = [load_image(m, max_side=1024) for m in (a.mood or [])]
        refs += [load_preset(n) for n in (a.preset or [])]
    except (FileNotFoundError, ValueError, KeyError) as e:
        log.error(str(e))
        return 2
    if a.weights and len(a.weights) != len(refs):
        log.error("--weights needs one value per reference (%d given, %d refs)", len(a.weights), len(refs))
        return 2
    t0 = time.perf_counter()
    out, info = transfer(content, refs, opt, device)
    sync(device)
    dt = time.perf_counter() - t0
    path = save_image(out, a.out)
    log.info("Saved %s (%dx%d) in %.2fs", path, out.shape[1], out.shape[0], dt)
    if "loss_history" in info:
        from src.gatys import plot_loss_curves
        lp = plot_loss_curves(info["loss_history"], Path(a.out).with_name(Path(a.out).stem + "_loss.png"))
        log.info("Gatys loss curve: %s (final total loss %.4g)", lp, info["loss_history"]["total"][-1])
    if a.describe:
        print(json.dumps({"reference": describe_mood(refs[0]).as_dict(),
                          "output": describe_mood(out).as_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
