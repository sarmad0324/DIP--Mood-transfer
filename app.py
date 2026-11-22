#!/usr/bin/env python
"""Gradio demo for Mood Transfer.

Launch:  python app.py            (then open http://127.0.0.1:7860)
         python app.py --share    (temporary public link)
"""
from __future__ import annotations

import argparse
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr
import numpy as np

from src.mood import (MODES, PostConfig, describe_markdown, describe_mood, list_presets, load_preset,
                      palette_image)
from src.pipeline import METHOD_LABELS, TransferOptions, transfer
from src.utils import (default_size, get_device, logger, resize_max_side, save_image, set_seed,
                       setup_logging, sync, to_float_rgb)

setup_logging()
DEVICE = get_device()
PRESETS = list_presets()
NONE = "(none — use uploaded image)"
LABEL_TO_METHOD = {v: k for k, v in METHOD_LABELS.items()}
MODE_LABELS = {"color": "color-only mood (keep luminance)", "full": "full mood",
               "tone": "tone-only mood (keep colours)", "color_tone": "color + tone (photographic)"}
LABEL_TO_MODE = {v: k for k, v in MODE_LABELS.items()}
MAX_UPLOAD_SIDE = 4096


def _prep(img: np.ndarray | None, what: str) -> np.ndarray | None:
    """Validate and normalise an uploaded image (numpy from Gradio)."""
    if img is None:
        return None
    try:
        img = to_float_rgb(img)
    except ValueError as e:
        raise gr.Error(f"{what}: {e}")
    if min(img.shape[:2]) < 16:
        raise gr.Error(f"{what} is too small ({img.shape[1]}x{img.shape[0]}).")
    if max(img.shape[:2]) > MAX_UPLOAD_SIDE:
        gr.Warning(f"{what} is very large; downscaling to {MAX_UPLOAD_SIDE}px long side.")
        img = resize_max_side(img, MAX_UPLOAD_SIDE)
    return img


def _reference(upload, preset, what: str) -> np.ndarray | None:
    ref = _prep(upload, what)
    if ref is None and preset and preset != NONE:
        ref = load_preset(preset)
    return ref


def preview_reference(upload, preset):
    """Show the chosen reference and its mood descriptor as soon as it changes."""
    try:
        ref = _reference(upload, preset, "Reference")
    except Exception:
        return None, None, ""
    if ref is None:
        return None, None, ""
    d = describe_mood(ref)
    return ref, palette_image(d), describe_markdown(d)


def run(content, ref_up, preset, ref2_up, preset2, blend_w, method_label, alpha, mode_label, refine,
        refine_method, size, gatys_iters, hist_refine, smooth, sharpen, grain, vignette, seed):
    t0 = time.perf_counter()
    c = _prep(content, "Content image")
    if c is None:
        raise gr.Error("Please upload a content image.")
    ref = _reference(ref_up, preset, "Reference")
    if ref is None:
        raise gr.Error("Upload a reference image or choose a mood preset.")
    refs, weights = [ref], None
    ref2 = _reference(ref2_up, preset2, "Second reference")
    if ref2 is not None and blend_w > 0:
        refs.append(ref2)
        weights = [1.0 - blend_w, blend_w]
    method = LABEL_TO_METHOD[method_label]
    size = int(size)
    if method == "gatys" and DEVICE.type == "cpu" and size > 256:
        gr.Warning("Gatys on CPU is slow: limiting size to 256 px.")
        size = 256
    set_seed(int(seed))
    opt = TransferOptions(method=method, alpha=float(alpha), mode=LABEL_TO_MODE[mode_label],
                          refine=bool(refine), refine_method=refine_method, size=size,
                          weights=weights, gatys_iters=int(gatys_iters),
                          post=PostConfig(hist_refine, smooth, sharpen, grain, vignette, int(seed)))
    try:
        out, _ = transfer(c, refs, opt, DEVICE)
        sync(DEVICE)
    except RuntimeError as e:  # e.g. GPU out-of-memory
        logger.error(traceback.format_exc())
        raise gr.Error(f"Transfer failed ({e}). Try a smaller output size.")
    dt = time.perf_counter() - t0
    path = Path(tempfile.mkdtemp()) / f"mood_{method}.png"
    save_image(out, path)
    d_out = describe_mood(out)
    info = (f"**{method_label}** on `{DEVICE}` — {out.shape[1]}×{out.shape[0]} px in **{dt:.2f}s**  \n"
            + describe_markdown(d_out).replace("**Mood:**", "**Output mood:**"))
    return resize_max_side(c, size), out, str(path), palette_image(d_out), info


def build() -> gr.Blocks:
    gpu = DEVICE.type != "cpu"
    with gr.Blocks(title="Mood Transfer") as demo:
        gr.Markdown(
            "# Mood Transfer\n"
            "Transfer the **mood** (colour, tone, lighting, atmosphere) of a reference photo onto your "
            "image with deep learning (pretrained VGG-19 + AdaIN), photorealistic guided-filter refinement "
            f"and classical baselines. Running on **{DEVICE}**."
            + ("" if gpu else "  \n⚠️ No GPU detected: Gatys is slow on CPU (keep size ≤ 256)."))
        with gr.Row():
            with gr.Column(scale=1):
                content = gr.Image(label="Content image", type="numpy", height=260)
                with gr.Tab("Mood reference"):
                    preset = gr.Dropdown([NONE] + list(PRESETS), value=(list(PRESETS) or [NONE])[0],
                                         label="Mood preset (ignored if you upload a reference)")
                    ref_up = gr.Image(label="…or upload a reference image", type="numpy", height=200)
                with gr.Tab("Blend a 2nd mood"):
                    preset2 = gr.Dropdown([NONE] + list(PRESETS), value=NONE, label="Second mood preset")
                    ref2_up = gr.Image(label="…or upload a second reference", type="numpy", height=160)
                    blend_w = gr.Slider(0, 1, 0.5, step=0.05, label="Weight of 2nd mood (0 = off)")
            with gr.Column(scale=1):
                method = gr.Dropdown(list(METHOD_LABELS.values()), value=METHOD_LABELS["adain_mood"],
                                     label="Method")
                alpha = gr.Slider(0, 1, 1.0, step=0.05, label="Strength α")
                mode = gr.Radio(list(MODE_LABELS.values()), value=MODE_LABELS["color"], label="Transfer mode")
                with gr.Row():
                    refine = gr.Checkbox(True, label="Photorealistic refinement")
                    refine_method = gr.Radio(["guided", "bilateral"], value="guided", label="Refinement filter")
                size = gr.Slider(128, 1024, default_size(DEVICE), step=32, label="Output size (long side, px)")
                gatys_iters = gr.Slider(50, 500, 200 if gpu else 100, step=25,
                                        label="Gatys iterations (slow method)")
                with gr.Accordion("Post-processing", open=False):
                    hist_refine = gr.Slider(0, 1, 0, step=0.05, label="Colour-histogram refinement")
                    smooth = gr.Slider(0, 1, 0, step=0.05, label="Edge-preserving smoothing")
                    sharpen = gr.Slider(0, 2, 0, step=0.05, label="Sharpen")
                    grain = gr.Slider(0, 1, 0, step=0.05, label="Film grain")
                    vignette = gr.Slider(0, 1, 0, step=0.05, label="Vignette")
                    seed = gr.Number(0, label="Seed", precision=0)
                go = gr.Button("Transfer mood", variant="primary")
        with gr.Row():
            out_content = gr.Image(label="Content", interactive=False, height=320)
            out_ref = gr.Image(label="Reference", interactive=False, height=320)
            out_img = gr.Image(label="Output", interactive=False, height=320)
        with gr.Row():
            with gr.Column():
                ref_pal = gr.Image(label="Reference palette", interactive=False, height=80)
                ref_md = gr.Markdown()
            with gr.Column():
                out_pal = gr.Image(label="Output palette", interactive=False, height=80)
                out_md = gr.Markdown()
        download = gr.File(label="Download result (PNG)")

        def on_method(m):
            # sensible mode/refinement defaults per method
            is_mood = m == METHOD_LABELS["adain_mood"]
            return (gr.update(value=MODE_LABELS["color" if is_mood else "full"]), gr.update(value=is_mood))

        method.change(on_method, method, [mode, refine])
        for comp in (ref_up, preset):
            comp.change(preview_reference, [ref_up, preset], [out_ref, ref_pal, ref_md])
        demo.load(preview_reference, [ref_up, preset], [out_ref, ref_pal, ref_md])
        go.click(run, [content, ref_up, preset, ref2_up, preset2, blend_w, method, alpha, mode, refine,
                       refine_method, size, gatys_iters, hist_refine, smooth, sharpen, grain, vignette, seed],
                 [out_content, out_img, download, out_pal, out_md])
        examples = sorted((Path(__file__).parent / "data" / "content").glob("*.jpg"))[:4]
        if examples:
            gr.Examples([[str(p)] for p in examples], inputs=[content], label="Example content images")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build().queue().launch(server_name="127.0.0.1", server_port=a.port, share=a.share)
