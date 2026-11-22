# PROGRESS LOG — Mood Transfer DIP Project

> If the session restarts, read this file first and resume from "Next action".

## Environment (inspected 2026-11-22)
- macOS (Darwin 25.6), Apple M5, 16 GB RAM, **no CUDA** — Apple **MPS** GPU used (auto-detect CUDA > MPS > CPU).
- Python 3.13.5, venv `mood-transfer/.venv`; torch 2.15.0, torchvision 0.29.2, gradio 6.32.0.
- Network quirks: `commons.wikimedia.org` TLS stalls and Python urllib is slow here → fetcher uses the en.wikipedia API (shared Commons metadata) via system `curl -4`.
- No pandoc; WeasyPrint lacks pango → PDF via headless Chrome (`scripts/export_pdf.py`).

## Steps completed (all tiers)
- [x] P0.1 Env, repo structure, venv, deps (installed in groups; a one-shot install was OOM-killed).
- [x] P0.2 `scripts/download_weights.py` — all weights downloaded & verified.
- [x] P0.3 utils / models / AdaIN — pretrained weights work; α=0 reconstruction verified visually.
- [x] P1.4 classical baselines; P1.5 Gatys (L-BFGS, loss curves).
- [x] P1.6 10 content + 6 mood images (Wikimedia Commons, CC0/BY/BY-SA) + CREDITS.md; extra NASA PD image for semantic demo.
- [x] P1.7 metrics + run_experiments; P1.8 Gradio app (verified end to end via gradio_client + Playwright/Chrome, screenshot saved).
- [x] P2.9 mood.py (modes, colour guided filter from scratch, descriptors, presets, blending, post-processing, semantic DeepLabV3).
- [x] P2.10 Full experiments re-run incl. AdaIN+mood (+ color+tone variant), Gatys subset 3×6 @384px/300 it., CPU+MPS speed benchmark, all figures.
- [x] P3.11 Tests (28 pass, ~3 s), README, report.md + report.pdf (21 pages), slides outline, viva prep (27 Q&A), notebook executed (0 errors).
- [x] P3.12 Final audit → CHECKLIST.md. Clean install in a fresh venv: `pip check` OK, 28 tests pass.

## Current working state
- COMPLETE and demonstrable. `python app.py` (GUI), `python run.py ...` (CLI), `python scripts/run_experiments.py --speed` (all results).

## Key results (real)
- AdaIN+mood (color): SSIM 0.996, LPIPS 0.006, Lab dist 26.8 (content 50.1), ΔCLIP-text 0.086, 0.10 s/img MPS.
- AdaIN+mood (color+tone): SSIM 0.757 (0.86 excl. noir), Lab 16.8, ΔCLIP-text 0.108 (best).
- AdaIN: SSIM 0.331; Reinhard SSIM 0.809 / Lab 6.9; Gatys (subset) SSIM 0.638, 20.6 s/img MPS, 57.9 s CPU.

## Known issues / deviations
- See CHECKLIST.md "Known deviations" (no CUDA; train_adain.py not needed; semantic = VOC fg/bg; AdaIN α=0 test at 192 px; PDF ~33 MB).

## Next action
- None required. Optional: compress report.pdf figures; run a user study with the Appendix A template.
