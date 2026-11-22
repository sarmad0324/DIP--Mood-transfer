# Slides outline — Mood Transfer (10 slides, ~10–12 min + demo)

**Slide 1 — Title**
* Mood Transfer: transferring the look, feel and atmosphere of one photo to another with deep learning
* Team / course / date. Teaser: content | golden-hour reference | our output (`figures/qualitative_grid.png`, row 1)

**Slide 2 — Problem & what "mood" means**
* Input A (structure to keep) + input B (mood to borrow) → A with B's colour, tone, lighting
* Mood = palette, colour temperature, brightness/contrast (tone curve), saturation, haze; *not* brush texture
* Requirements: deep learning core, content preserved, no artefacts, photographic look
* Visual: the 6 presets with palettes and labels (`figures/mood_presets_descriptors.png`)

**Slide 3 — Background (DIP → DL)**
* Classical: Reinhard (Lab mean/std), histogram specification (CDF matching): global, blind to structure
* Gatys: VGG-19 content loss plus Gram-matrix style loss, optimise pixels
* AdaIN: `σ(y)(x−μ(x))/σ(x)+μ(y)` on deep features, which is "Reinhard on 512 VGG channels" + a decoder
* Photorealistic transfer (Luan et al., PhotoWCT): locally affine colour, smoothing

**Slide 4 — Our pipeline**
* Diagram `figures/architecture_adain_mood.png`
* Pretrained VGG-19 encoder → AdaIN (α, multi-reference μ/σ mix) → pretrained decoder
* Mood layer: Lab colour-only mode (L* from content) → colour guided filter (guide = content) → optional post-processing
* Encoder must match the decoder (normalised VGG, raw RGB input) vs torchvision VGG for Gatys

**Slide 5 — Why plain AdaIN fails on photos, and the fix**
* `figures/refinement_before_after.png`: raw AdaIN (painterly, warped) → colour mode → + guided filter
* Guided filter: q = aᵀI + b per window ⇒ edges only where the content has edges
* Ablation `figures/guided_filter_ablation.png`: r = long/16, ε = 1e-2 removes chroma speckle

**Slide 6 — Controls: modes, α, blending**
* `figures/transfer_modes.png` (full / colour-only / tone-only / colour+tone)
* `figures/alpha_sweep.png` (α = 0 … 1)
* `figures/multi_reference_blending.png` (golden hour ↔ blue dusk)

**Slide 7 — Quantitative results**
* Table from `results/metrics_summary.md` (10 × 6 pairs; Gatys on a documented 3 × 6 subset)
* `figures/metrics_summary.png` and `figures/tradeoff_scatter.png`
* Message: structure (SSIM/LPIPS) vs mood match trade-off
* Key numbers: ours SSIM **0.996** / LPIPS **0.006** vs AdaIN 0.331 / 0.477, Gatys 0.638 / 0.265;
  Lab-stat distance 50.1 → 26.8 (colour mode) → 16.8 (colour+tone); ΔCLIP-text: colour+tone **0.108** (best),
  Reinhard 0.089; speed: ours **0.10 s** (MPS) vs Gatys **20.6 s**, ~200× faster

**Slide 8 — Qualitative comparison**
* `figures/qualitative_grid.png`: Content | Ref | Reinhard | Hist | Gatys | AdaIN | Ours

**Slide 9 — Failure cases & limitations**
* `figures/failure_cases.png`: RGB histogram colour cast, AdaIN texture leakage, Reinhard scene mismatch,
  over-smoothed filter halos, our residual colour blob
* Global statistics can't relight a scene; colour-only mode under-transfers brightness-driven moods (noir)
* Semantic extension (`figures/semantic_transfer.png`): VOC has no "sky" class

**Slide 10 — Demo & conclusion**
* Live Gradio demo (`python app.py`): preset → method → α → mode → post-processing → download
* CLI + batch, tests, reproducible `run_experiments.py`
* Future: semantic correspondence (sky↔sky), learned photorealistic decoders (WCT²), video consistency, user study
