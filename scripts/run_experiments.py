"""Generate ALL results: outputs, metrics.csv, summary plots and report figures.

Protocol
--------
* Reinhard, Histogram (Lab), AdaIN, AdaIN + mood enhancements: every content × every mood
  (≥10 × ≥6) at ``--size`` (default 512 on GPU / 384 on CPU).
* Gatys (slow): documented subset — the first ``--gatys-contents`` contents (sorted by
  filename) × all moods, at ``--gatys-size`` px and ``--gatys-iters`` L-BFGS evaluations.
* An "identity" row (output = content) is included as a reference point for every metric.
* Speed: measured per image on the main device; ``--speed`` additionally benchmarks
  CPU vs GPU on a few pairs.

Usage:
    python scripts/run_experiments.py                  # everything
    python scripts/run_experiments.py --skip-gatys     # fast pass
    python scripts/run_experiments.py --figures-only   # re-make figures from saved outputs
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.adain import get_adain  # noqa: E402
from src.classical import histogram_matching, reinhard_transfer  # noqa: E402
from src.gatys import GatysConfig, get_gatys, plot_loss_curves  # noqa: E402
from src.metrics import ClipScorer, all_metrics  # noqa: E402
from src.mood import (PRESET_NAMES, MoodConfig, PostConfig, adain_mood_transfer, apply_mode,  # noqa: E402
                      describe_mood, palette_image, semantic_mood_transfer)
from src.pipeline import METHOD_LABELS  # noqa: E402
from src.utils import (DATA_DIR, RESULTS_DIR, default_size, get_device, list_images, load_image,  # noqa: E402
                       resize_max_side, resize_to, save_image, set_seed, setup_logging, sync)

OUT = RESULTS_DIR / "outputs"
FIG = RESULTS_DIR / "figures"
REPORT_FIG = ROOT / "report" / "figures"
FAST_METHODS = ("reinhard", "hist", "adain", "adain_mood", "adain_mood_ct")
COL_LABELS = {"identity": "Identity (content)", **METHOD_LABELS,
              "adain_mood_ct": "AdaIN + mood (color+tone mode)"}
#: text prompts for CLIP image–text similarity, keyed by mood file stem
MOOD_PROMPTS = {
    "golden_hour": "a photo in warm golden hour sunset light",
    "blue_dusk": "a photo at cold blue dusk, blue hour",
    "foggy": "a foggy, misty, hazy photo",
    "neon_night": "a photo at night lit by colorful neon lights",
    "vintage_film": "a faded vintage film photograph",
    "dark_noir": "a dark, moody, low-key noir photo",
}
log = setup_logging()


# ======================================================================================
# helpers
# ======================================================================================
def run_method(method: str, c: np.ndarray, r: np.ndarray, size: int, device, gatys_cfg=None):
    """Run a method at working size; return (output, seconds, extra)."""
    c = resize_max_side(c, size)
    r = resize_max_side(r, size)
    sync(device)
    t0 = time.perf_counter()
    extra = {}
    if method == "reinhard":
        out = reinhard_transfer(c, r)
    elif method == "hist":
        out = histogram_matching(c, r, "lab")
    elif method == "adain":
        out = get_adain(device).stylize(c, r, alpha=1.0, size=None)
    elif method == "adain_mood":
        out = adain_mood_transfer(c, r, MoodConfig(size=None), device)
    elif method == "adain_mood_ct":
        out = adain_mood_transfer(c, r, MoodConfig(size=None, mode="color_tone"), device)
    elif method == "gatys":
        out, extra["hist"] = get_gatys(device).stylize(c, r, gatys_cfg, progress=False)
    elif method == "identity":
        out = c.copy()
    else:
        raise ValueError(method)
    sync(device)
    return out, time.perf_counter() - t0, extra


def out_path(method: str, cname: str, mname: str) -> Path:
    return OUT / method / f"{cname}__{mname}.png"


def label_img(ax, img, title=None, fs=9):
    ax.imshow(np.clip(img, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fs)


def savefig(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, REPORT_FIG / name)
    log.info("figure: %s", p.relative_to(ROOT))


def mood_title(stem: str) -> str:
    return PRESET_NAMES.get(stem, stem.replace("_", " ").title())


# ======================================================================================
# main evaluation
# ======================================================================================
def evaluate(contents, moods, args, device) -> pd.DataFrame:
    clip = ClipScorer(str(device))
    rows = []
    gatys_cfg = GatysConfig(iterations=args.gatys_iters, size=args.gatys_size)
    gatys_contents = [p.stem for p in contents[: args.gatys_contents]]
    # warm-up (first call includes weight loading / kernel compilation)
    _c, _r = load_image(contents[0], 256), load_image(moods[0], 256)
    for m in FAST_METHODS:
        run_method(m, _c, _r, 256, device)
    text_emb = {k: clip.embed_text(v) for k, v in MOOD_PROMPTS.items()}

    methods = tuple(args.only) if args.only else ("identity",) + FAST_METHODS
    jobs = [(cp, mp, m) for cp in contents for mp in moods for m in methods]
    if not args.skip_gatys and not args.only:
        jobs += [(cp, mp, "gatys") for cp in contents if cp.stem in gatys_contents for mp in moods]
    cache_c, cache_m, clip_cr = {}, {}, {}
    for cp, mp, method in tqdm(jobs, desc="experiments"):
        if cp not in cache_c:
            cache_c[cp] = load_image(cp, 2048)
        if mp not in cache_m:
            cache_m[mp] = load_image(mp, 1024)
        size = args.gatys_size if method == "gatys" else args.size
        c = resize_max_side(cache_c[cp], size)
        r = resize_max_side(cache_m[mp], size)
        set_seed(0)
        out, secs, extra = run_method(method, c, r, size, device, gatys_cfg)
        save_image(out, out_path(method, cp.stem, mp.stem))
        if method == "gatys" and cp.stem == gatys_contents[0]:
            plot_loss_curves(extra["hist"], RESULTS_DIR / "gatys_loss" / f"{cp.stem}__{mp.stem}.png",
                             f"Gatys losses: {cp.stem} × {mp.stem}")
        key = (cp.stem, mp.stem, size)
        if key not in clip_cr:
            clip_cr[key] = clip.image_similarity(c, r)
        met = all_metrics(c, out, r, clip, clip_cr[key], device=str(device))
        e = clip.embed_image(out)
        if mp.stem in text_emb:
            met["clip_text"] = float((e * text_emb[mp.stem]).sum())
            met["clip_text_gain"] = met["clip_text"] - float((clip.embed_image(c) * text_emb[mp.stem]).sum())
        rows.append({"content": cp.stem, "mood": mp.stem, "method": method, "size": size,
                     "subset": "gatys_subset" if method == "gatys" else "full",
                     "seconds": 0.0 if method == "identity" else secs, "device": str(device), **met})
    df = pd.DataFrame(rows)
    if args.only and (RESULTS_DIR / "metrics.csv").exists():   # merge into the existing table
        old = pd.read_csv(RESULTS_DIR / "metrics.csv")
        df = pd.concat([old[~old.method.isin(args.only)], df], ignore_index=True)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    log.info("wrote results/metrics.csv (%d rows)", len(df))
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per method; Gatys is also compared on the identical subset."""
    cols = ["ssim", "lpips", "lpips_rgb", "lab_stat_dist", "wasserstein_lab", "bhattacharyya",
            "clip_sim", "clip_gain", "clip_text", "clip_text_gain", "seconds"]
    cols = [c for c in cols if c in df]
    order = [m for m in ("identity",) + FAST_METHODS + ("gatys",) if m in set(df.method)]
    parts = []
    # (a) full grid
    full = df[df.method != "gatys"]
    g = full.groupby("method")[cols].agg(["mean", "std"]).reindex([m for m in order if m != "gatys"])
    g.insert(0, ("n", ""), full.groupby("method").size())
    g.insert(0, ("eval_set", ""), "full grid")
    parts.append(g)
    # (b) Gatys subset: every method restricted to the same contents
    if "gatys" in set(df.method):
        sub_contents = set(df[df.method == "gatys"].content)
        sub = df[df.content.isin(sub_contents)]
        g2 = sub.groupby("method")[cols].agg(["mean", "std"]).reindex(order)
        g2.insert(0, ("n", ""), sub.groupby("method").size())
        g2.insert(0, ("eval_set", ""), "gatys subset")
        g2.index = [f"{i} [subset]" for i in g2.index]
        parts.append(g2)
    s = pd.concat(parts)
    s.columns = [f"{a}_{b}" if b else a for a, b in s.columns]
    s.to_csv(RESULTS_DIR / "metrics_summary.csv")
    # compact markdown table for the README/report
    md = ["| Method | Eval set | n | SSIM ↑ | LPIPS ↓ | Lab stat dist ↓ | Wasserstein ↓ | Bhattacharyya ↓ | CLIP img ↑ | ΔCLIP img ↑ | ΔCLIP text ↑ | s/img |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for idx, r in s.iterrows():
        name = COL_LABELS.get(idx.replace(" [subset]", ""), idx)
        def f(k, p=3):
            return f"{r[k + '_mean']:.{p}f} ± {r[k + '_std']:.{p}f}" if k + "_mean" in r else "–"
        md.append(f"| {name} | {r['eval_set']} | {int(r['n'])} | {f('ssim')} | {f('lpips')} | "
                  f"{f('lab_stat_dist', 1)} | {f('wasserstein_lab', 2)} | {f('bhattacharyya')} | "
                  f"{f('clip_sim')} | {f('clip_gain')} | {f('clip_text_gain')} | {r['seconds_mean']:.3f} |")
    (RESULTS_DIR / "metrics_summary.md").write_text("\n".join(md) + "\n")
    log.info("wrote results/metrics_summary.csv/.md")
    return s


def plot_summary(df: pd.DataFrame):
    metrics = [("ssim", "SSIM (luma) vs content ↑"), ("lpips", "LPIPS (luma) vs content ↓"),
               ("lab_stat_dist", "Lab mean/std distance to ref ↓"),
               ("wasserstein_lab", "Wasserstein (Lab) to ref ↓"),
               ("clip_gain", "ΔCLIP image sim. to ref ↑"), ("seconds", "Seconds / image (log)")]
    order = [m for m in ("identity",) + FAST_METHODS + ("gatys",) if m in set(df.method)]
    colors = {"identity": "#9e9e9e", "reinhard": "#8da0cb", "hist": "#66c2a5", "gatys": "#e78ac3",
              "adain": "#fc8d62", "adain_mood": "#d62728", "adain_mood_ct": "#8c1515"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (k, title) in zip(axes.ravel(), metrics):
        ms = [m for m in order if not (k == "seconds" and m == "identity")]
        means = [df[df.method == m][k].mean() for m in ms]
        stds = [df[df.method == m][k].std() for m in ms]
        ax.bar(range(len(ms)), means, yerr=stds, capsize=4, color=[colors[m] for m in ms])
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([COL_LABELS[m].replace(" + mood enhancements", "\n+ mood").replace(" (content)", "")
                            .replace(" (color+tone mode)", "\n(col+tone)")
                            + ("\n(subset)" if m == "gatys" else "") for m in ms], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if k == "seconds":
            ax.set_yscale("log")
    fig.suptitle("Mean ± std per method (Gatys on its documented subset)", fontsize=12)
    fig.tight_layout()
    savefig(fig, "metrics_summary.png")

    # trade-off scatter: content preservation vs mood match
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for m in order:
        d = df[df.method == m]
        ax.scatter(d.lab_stat_dist, d.ssim, s=12, alpha=0.35, color=colors[m])
        ax.scatter(d.lab_stat_dist.mean(), d.ssim.mean(), s=140, color=colors[m], edgecolor="k",
                   label=COL_LABELS[m])
    ax.set_xlabel("Lab mean/std distance to reference (↓ better mood match)")
    ax.set_ylabel("SSIM to content (↑ better structure)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Structure vs mood trade-off (each dot = one pair)")
    savefig(fig, "tradeoff_scatter.png")


# ======================================================================================
# figures
# ======================================================================================
def fig_qualitative(contents, moods, pairs, size):
    cols = ["content", "ref", "reinhard", "hist", "gatys", "adain", "adain_mood"]
    titles = ["Content", "Mood reference", "Reinhard", "Hist-match (Lab)", "Gatys", "AdaIN",
              "AdaIN + mood\n(ours)"]
    fig, axes = plt.subplots(len(pairs), len(cols), figsize=(2.6 * len(cols), 2.0 * len(pairs) + 0.5))
    for i, (cn, mn) in enumerate(pairs):
        c = load_image(DATA_DIR / "content" / f"{cn}.jpg", size)
        for j, col in enumerate(cols):
            if col == "content":
                img = c
            elif col == "ref":
                img = load_image(_find(moods, mn), size)
            else:
                p = out_path(col, cn, mn)
                img = load_image(p) if p.exists() else np.ones_like(c)
            label_img(axes[i, j], img, titles[j] if i == 0 else None, 10)
        axes[i, 0].set_ylabel(mood_title(mn), fontsize=9)
    fig.tight_layout()
    savefig(fig, "qualitative_grid.png")


def _find(paths, stem):
    return next(p for p in paths if p.stem == stem)


def fig_alpha_sweep(c, r, device, size):
    alphas = [0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(2, len(alphas) + 1, figsize=(2.8 * (len(alphas) + 1), 5.2))
    c = resize_max_side(c, size)
    label_img(axes[0, 0], c, "Content"); label_img(axes[1, 0], resize_max_side(r, size), "Reference")
    for j, a in enumerate(alphas):
        o1 = get_adain(device).stylize(c, r, alpha=a, size=size)
        o2 = adain_mood_transfer(c, r, MoodConfig(alpha=a, size=size), device)
        label_img(axes[0, j + 1], o1, f"AdaIN α={a}")
        label_img(axes[1, j + 1], o2, f"AdaIN+mood α={a}")
        save_image(o2, RESULTS_DIR / "alpha_sweep" / f"adain_mood_alpha{a}.png")
        save_image(o1, RESULTS_DIR / "alpha_sweep" / f"adain_alpha{a}.png")
    fig.tight_layout()
    savefig(fig, "alpha_sweep.png")


def fig_modes(c, r, device, size):
    c = resize_max_side(c, size)
    st = get_adain(device).stylize(c, r, alpha=1.0, size=size)
    panels = [("Content", c), ("Reference", resize_max_side(r, size)),
              ("full mood (raw AdaIN)", st)]
    for mode, name in (("color", "color-only mood"), ("tone", "tone-only mood"),
                       ("color_tone", "color + tone")):
        panels.append((name + "\n(+ refinement)", apply_mode(c, st, mode, refine=True)))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 2.9))
    for ax, (t, im) in zip(axes, panels):
        label_img(ax, im, t)
    fig.tight_layout()
    savefig(fig, "transfer_modes.png")


def fig_refinement(c, r, device, size, crop_frac=(0.25, 0.25, 0.35)):
    """Before/after the guided-filter photorealistic refinement, with zoomed crops."""
    c = resize_max_side(c, size)
    st = get_adain(device).stylize(c, r, alpha=1.0, size=size)
    variants = [("Content", c), ("Raw AdaIN (full)", st),
                ("color mode, NO refinement", apply_mode(c, st, "color", refine=False)),
                ("color mode + guided filter", apply_mode(c, st, "color", refine=True)),
                ("full mode + guided filter", apply_mode(c, st, "full", refine=True))]
    h, w = c.shape[:2]
    y0, x0, s = int(crop_frac[0] * h), int(crop_frac[1] * w), int(crop_frac[2] * min(h, w))
    fig, axes = plt.subplots(2, len(variants), figsize=(3 * len(variants), 6))
    for j, (t, im) in enumerate(variants):
        label_img(axes[0, j], im, t)
        axes[0, j].add_patch(plt.Rectangle((x0, y0), s, s, fill=False, ec="yellow", lw=1.5))
        label_img(axes[1, j], resize_to(im[y0:y0 + s, x0:x0 + s], (256, 256)), "zoom")
    fig.tight_layout()
    savefig(fig, "refinement_before_after.png")
    from src.metrics import ssim_luma
    return {t: round(ssim_luma(c, im), 4) for t, im in variants}


def fig_blending(c, r1, r2, n1, n2, device, size):
    ws = [0.0, 0.25, 0.5, 0.75, 1.0]
    c = resize_max_side(c, size)
    fig, axes = plt.subplots(1, len(ws) + 2, figsize=(2.7 * (len(ws) + 2), 2.9))
    label_img(axes[0], resize_max_side(r1, size), f"Ref A: {n1}")
    for j, w in enumerate(ws):
        o = adain_mood_transfer(c, [r1, r2], MoodConfig(size=size, weights=[1 - w, w]), device)
        label_img(axes[j + 1], o, f"{int(100 * (1 - w))}% A / {int(100 * w)}% B")
    label_img(axes[-1], resize_max_side(r2, size), f"Ref B: {n2}")
    fig.tight_layout()
    savefig(fig, "multi_reference_blending.png")


def fig_presets(moods):
    fig, axes = plt.subplots(2, len(moods), figsize=(2.8 * len(moods), 3.6),
                             gridspec_kw={"height_ratios": [3, 1]})
    desc = {}
    for j, mp in enumerate(moods):
        im = load_image(mp, 512)
        d = describe_mood(im)
        desc[mp.stem] = d.as_dict()
        label_img(axes[0, j], im, mood_title(mp.stem))
        label_img(axes[1, j], palette_image(d, 300, 60))
        axes[1, j].set_xlabel(d.label.replace(", ", ",\n", 1), fontsize=8)
    fig.tight_layout()
    savefig(fig, "mood_presets_descriptors.png")
    (RESULTS_DIR / "mood_descriptors.json").write_text(json.dumps(desc, indent=2))


def fig_postprocessing(c, r, device, size):
    c = resize_max_side(c, size)
    base = adain_mood_transfer(c, r, MoodConfig(size=size), device)
    cfgs = [("no post-processing", PostConfig()), ("hist. refinement 0.6", PostConfig(hist_refine=0.6)),
            ("sharpen 0.8", PostConfig(sharpen=0.8)), ("grain 0.5 + vignette 0.5", PostConfig(grain=0.5, vignette=0.5)),
            ("bilateral smooth 0.7", PostConfig(smooth=0.7))]
    from src.mood import post_process
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3 * len(cfgs), 3))
    for ax, (t, cfg) in zip(axes, cfgs):
        label_img(ax, post_process(base, r, cfg), t)
    fig.tight_layout()
    savefig(fig, "post_processing.png")


def fig_failures(cases, contents, moods, device, size):
    """Failure-case panels: each case = (title, content stem, mood stem, method/variant)."""
    fig, axes = plt.subplots(len(cases), 3, figsize=(9, 2.9 * len(cases)))
    for i, (title, cn, mn, variant) in enumerate(cases):
        c = load_image(_find(contents, cn), size)
        r = load_image(_find(moods, mn), size)
        if variant == "hist_rgb":
            out = histogram_matching(c, r, "rgb")
        elif variant == "reinhard":
            out = reinhard_transfer(c, r)
        elif variant == "adain":
            out = get_adain(device).stylize(c, r, alpha=1.0, size=size)
        elif variant == "full_refine":
            out = apply_mode(c, get_adain(device).stylize(c, r, size=size), "full", refine=True)
        elif variant == "adain_mood":
            out = adain_mood_transfer(c, r, MoodConfig(size=size), device)
        elif variant == "big_radius":
            st = get_adain(device).stylize(c, r, size=size)
            from src.mood import guided_filter_color
            from src.utils import lab_to_rgb, rgb_to_lab
            lab = rgb_to_lab(c)
            lab[..., 1:] = rgb_to_lab(np.clip(guided_filter_color(c, st, 96, 0.2), 0, 1))[..., 1:]
            out = lab_to_rgb(lab)
        else:
            raise ValueError(variant)
        label_img(axes[i, 0], c, "Content" if i == 0 else None)
        label_img(axes[i, 1], r, "Reference" if i == 0 else None)
        label_img(axes[i, 2], out, None)
        axes[i, 2].set_title(title, fontsize=9)
        save_image(out, RESULTS_DIR / "failures" / f"{i}_{variant}_{cn}__{mn}.png")
    fig.tight_layout()
    savefig(fig, "failure_cases.png")


def fig_semantic(c, r_bg, r_fg, device, size):
    c = resize_max_side(c, size)
    uni = adain_mood_transfer(c, r_bg, MoodConfig(size=size), device)
    sem, mask = semantic_mood_transfer(c, r_bg, r_fg, cfg=MoodConfig(size=size), device=device)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    label_img(axes[0], c, "Content")
    label_img(axes[1], np.repeat(mask[..., None], 3, 2), "DeepLabV3 soft fg mask")
    label_img(axes[2], uni, "Uniform mood")
    label_img(axes[3], sem, "Semantic: bg/fg moods")
    fig.tight_layout()
    savefig(fig, "semantic_transfer.png")
    save_image(sem, RESULTS_DIR / "semantic" / "semantic.png")


def fig_gf_ablation(pairs, device, size):
    """Ablation of the guided-filter radius r and regulariser ε (colour mode).

    Also reports, per setting, the mean SSIM to the content and the mean high-frequency
    chroma energy (std of a*,b* minus their Gaussian-blurred version) — a proxy for the
    speckle/texture that refinement should remove.
    """
    import cv2
    from src.metrics import ssim_luma
    from src.mood import guided_filter_color
    from src.utils import lab_to_rgb, rgb_to_lab
    settings = [("no refinement", None, None), ("r=8, ε=1e-3", 8, 1e-3), ("r=16, ε=1e-3", 16, 1e-3),
                ("r=32, ε=1e-3", 32, 1e-3), ("r=32, ε=1e-2 (default)", 32, 1e-2), ("r=64, ε=1e-2", 64, 1e-2)]
    fig, axes = plt.subplots(len(pairs), len(settings) + 1, figsize=(2.6 * (len(settings) + 1), 2.1 * len(pairs)))
    axes = np.atleast_2d(axes)
    stats = {t: {"ssim": [], "chroma_hf": []} for t, _, _ in settings}
    for i, (c, r) in enumerate(pairs):
        c = resize_max_side(c, size)
        st = get_adain(device).stylize(c, r, alpha=1.0, size=size)
        label_img(axes[i, 0], c, "Content" if i == 0 else None)
        for j, (t, rad, eps) in enumerate(settings):
            src = st if rad is None else np.clip(guided_filter_color(c, st, int(rad * size / 512), eps), 0, 1)
            lab = rgb_to_lab(c)
            lab[..., 1:] = rgb_to_lab(src)[..., 1:]
            out = lab_to_rgb(lab)
            ab = rgb_to_lab(out)[..., 1:]
            hf = float(np.mean([np.std(ab[..., k] - cv2.GaussianBlur(ab[..., k], (0, 0), 2)) for k in range(2)]))
            stats[t]["ssim"].append(ssim_luma(c, out)); stats[t]["chroma_hf"].append(hf)
            label_img(axes[i, j + 1], out, t if i == 0 else None, 8)
    fig.tight_layout()
    savefig(fig, "guided_filter_ablation.png")
    return {t: {k: float(np.mean(v)) for k, v in d.items()} for t, d in stats.items()}


def fig_gatys_loss():
    d = RESULTS_DIR / "gatys_loss"
    if d.exists():
        imgs = sorted(d.glob("*.png"))
        if imgs:
            shutil.copy(imgs[0], REPORT_FIG / "gatys_loss_curve.png")
            shutil.copy(imgs[0], FIG / "gatys_loss_curve.png")


def fig_architecture():
    """Block diagram of the AdaIN + mood pipeline (matplotlib, no external tools)."""
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=color, ec="k", lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3), zorder=1)
    box(0.1, 2.9, 1.3, 0.8, "Content c", "#e8f0fe"); box(0.1, 0.5, 1.3, 0.8, "Mood ref s\n(1..N refs)", "#fde8e8")
    box(2.0, 2.9, 1.6, 0.8, "VGG-19 enc.\n(frozen, →relu4_1)", "#fff3c4"); box(2.0, 0.5, 1.6, 0.8, "VGG-19 enc.\n(shared weights)", "#fff3c4")
    box(4.0, 1.6, 2.1, 1.1, "AdaIN\nσ(s)·(f−μ(f))/σ(f) + μ(s)\n(multi-ref: weighted\nμ, σ mix)", "#e6f4ea")
    box(6.3, 1.7, 1.3, 0.9, "α·t + (1−α)·f_c", "#e6f4ea")
    box(8.0, 1.7, 1.3, 0.9, "Decoder\n(pretrained)", "#fff3c4")
    box(9.7, 1.7, 1.5, 0.9, "Transfer mode\n(Lab: L from c,\nab from output)", "#ede7f6")
    box(11.5, 1.7, 1.4, 0.9, "Guided filter\n(guide = c)\n+ post-proc.", "#ede7f6")
    arrow(1.4, 3.3, 2.0, 3.3); arrow(1.4, 0.9, 2.0, 0.9)
    arrow(3.6, 3.3, 4.6, 2.7); arrow(3.6, 0.9, 4.6, 1.6)
    arrow(6.1, 2.15, 6.3, 2.15); arrow(7.6, 2.15, 8.0, 2.15); arrow(9.3, 2.15, 9.7, 2.15); arrow(11.2, 2.15, 11.5, 2.15)
    ax.annotate("", xy=(10.4, 2.6), xytext=(1.4, 3.6), arrowprops=dict(arrowstyle="->", lw=1, ls="--", color="gray"))
    ax.text(6.0, 3.75, "content luminance + guide image (skip connection)", fontsize=8, color="gray")
    ax.text(12.2, 1.4, "output", ha="center", fontsize=9, weight="bold")
    ax.text(0.1, 4.0, "Deep learning core (pretrained VGG-19 + AdaIN decoder)  →  DIP mood layer (colour space, guided filter)",
            fontsize=9, weight="bold")
    savefig(fig, "architecture_adain_mood.png")

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3); ax.axis("off")
    box(0.1, 1.9, 1.5, 0.7, "x (init = content)", "#e8f0fe"); box(0.1, 0.3, 1.5, 0.7, "content c / ref s", "#fde8e8")
    box(2.3, 1.1, 2.0, 0.9, "VGG-19 (torchvision,\nImageNet norm.)", "#fff3c4")
    box(5.0, 2.0, 2.4, 0.7, "L_content (relu4_2 MSE)", "#e6f4ea")
    box(5.0, 1.2, 2.4, 0.7, "L_style (Gram, relu1_1..5_1)", "#e6f4ea")
    box(5.0, 0.4, 2.4, 0.7, "L_TV (smoothness)", "#e6f4ea")
    box(8.1, 1.1, 2.6, 0.9, "L-BFGS update of x\n∇x L by backprop", "#ede7f6")
    arrow(1.6, 2.2, 2.3, 1.7); arrow(1.6, 0.7, 2.3, 1.4)
    for y in (2.35, 1.55, 0.75):
        arrow(4.3, 1.55, 5.0, y); arrow(7.4, y, 8.1, 1.55)
    ax.annotate("", xy=(0.9, 2.6), xytext=(9.4, 2.0), arrowprops=dict(arrowstyle="->", lw=1, ls="--",
                connectionstyle="arc3,rad=0.25", color="gray"))
    savefig(fig, "architecture_gatys.png")


def speed_benchmark(contents, moods, size, gatys_size, gatys_iters):
    """Seconds/image on CPU and on the GPU (if any) for every method."""
    import torch
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + \
              (["cuda"] if torch.cuda.is_available() else [])
    rows = []
    pairs = [(contents[i], moods[i % len(moods)]) for i in range(3)]
    for dev in map(torch.device, devices):
        for method in FAST_METHODS + ("gatys",):
            n = len(pairs)
            sz = gatys_size if method == "gatys" else size
            cfg = GatysConfig(iterations=gatys_iters, size=gatys_size)
            run_method(method, load_image(pairs[0][0], 128), load_image(pairs[0][1], 128), 128, dev,
                       GatysConfig(iterations=2, size=128))  # warm-up
            ts = []
            for cp, mp in pairs[:n]:
                _, s, _ = run_method(method, load_image(cp, 2048), load_image(mp, 1024), sz, dev, cfg)
                ts.append(s)
            rows.append({"device": str(dev), "method": method, "size": sz, "n": n,
                         "sec_mean": float(np.mean(ts)), "sec_std": float(np.std(ts)),
                         "note": f"{gatys_iters} L-BFGS evals" if method == "gatys" else ""})
            log.info("speed %s %s: %.3fs", dev, method, np.mean(ts))
    sp = pd.DataFrame(rows)
    sp.to_csv(RESULTS_DIR / "speed.csv", index=False)
    return sp


# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--gatys-contents", type=int, default=3)
    ap.add_argument("--gatys-size", type=int, default=384)
    ap.add_argument("--gatys-iters", type=int, default=300)
    ap.add_argument("--skip-gatys", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--only", nargs="+", help="(re)run only these methods and merge into metrics.csv")
    ap.add_argument("--speed", action="store_true", help="also run the CPU/GPU speed benchmark")
    ap.add_argument("--config", default=str(ROOT / "scripts" / "figure_config.json"),
                    help="curated pairs for figures")
    args = ap.parse_args()
    set_seed(0)
    device = get_device()
    args.size = args.size or default_size(device)
    log.info("device=%s size=%d", device, args.size)
    contents = list_images(DATA_DIR / "content")
    moods = list_images(DATA_DIR / "moods")
    log.info("%d contents × %d moods", len(contents), len(moods))
    assert len(contents) >= 10 and len(moods) >= 6, "need ≥10 contents and ≥6 moods"
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.figures_only:
        df = pd.read_csv(RESULTS_DIR / "metrics.csv")
    else:
        df = evaluate(contents, moods, args, device)
    summarize(df)
    plot_summary(df)
    if args.speed:
        speed_benchmark(contents, moods, args.size, args.gatys_size, args.gatys_iters)

    cfg = json.loads(Path(args.config).read_text()) if Path(args.config).exists() else {}
    gatys_stems = sorted(set(df[df.method == "gatys"].content)) or [contents[0].stem]
    pairs = cfg.get("qualitative_pairs") or [(gatys_stems[i % len(gatys_stems)], moods[i % len(moods)].stem)
                                             for i in range(5)]
    fig_qualitative(contents, moods, pairs, args.size)
    a = cfg.get("alpha_pair", [contents[0].stem, moods[0].stem])
    c, r = load_image(_find(contents, a[0]), 2048), load_image(_find(moods, a[1]), 1024)
    fig_alpha_sweep(c, r, device, args.size)
    m = cfg.get("modes_pair", a)
    fig_modes(load_image(_find(contents, m[0]), 2048), load_image(_find(moods, m[1]), 1024), device, args.size)
    rf = cfg.get("refine_pair", a)
    ref_ssim = fig_refinement(load_image(_find(contents, rf[0]), 2048), load_image(_find(moods, rf[1]), 1024),
                              device, args.size, tuple(cfg.get("refine_crop", (0.25, 0.25, 0.35))))
    (RESULTS_DIR / "refinement_ssim.json").write_text(json.dumps(ref_ssim, indent=2))
    b = cfg.get("blend", [contents[0].stem, moods[0].stem, moods[1].stem])
    fig_blending(load_image(_find(contents, b[0]), 2048), load_image(_find(moods, b[1]), 1024),
                 load_image(_find(moods, b[2]), 1024), mood_title(b[1]), mood_title(b[2]), device, args.size)
    fig_presets(moods)
    p = cfg.get("post_pair", a)
    fig_postprocessing(load_image(_find(contents, p[0]), 2048), load_image(_find(moods, p[1]), 1024), device, args.size)
    fails = cfg.get("failures") or [
        ("Hist-match RGB: colour cast / over-saturation", contents[1].stem, moods[3].stem, "hist_rgb"),
        ("Plain AdaIN: texture leakage, warped edges", contents[2].stem, moods[0].stem, "adain"),
        ("Reinhard: mismatched scene content", contents[3].stem, moods[1].stem, "reinhard"),
        ("Over-smoothed guided filter (r=96, ε=0.2): colour halos", contents[4].stem, moods[0].stem, "big_radius")]
    fig_failures([tuple(f) for f in fails], contents, moods, device, args.size)
    s = cfg.get("semantic", [contents[0].stem, moods[0].stem, moods[1].stem])
    sem_c = ROOT / s[0] if (ROOT / s[0]).exists() else _find(contents, s[0])
    fig_semantic(load_image(sem_c, 2048), load_image(_find(moods, s[1]), 1024),
                 load_image(_find(moods, s[2]), 1024), device, args.size)
    gfa = cfg.get("gf_ablation", [[contents[0].stem, moods[0].stem]])
    gf_stats = fig_gf_ablation([(load_image(_find(contents, a_), 2048), load_image(_find(moods, b_), 1024))
                                for a_, b_ in gfa], device, args.size)
    (RESULTS_DIR / "gf_ablation.json").write_text(json.dumps(gf_stats, indent=2))
    fig_gatys_loss()
    fig_architecture()
    log.info("done.")


if __name__ == "__main__":
    main()
