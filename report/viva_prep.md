# Viva preparation — Mood Transfer

Short, defensible answers. Numbers quoted are from our own runs (`results/metrics_summary.md`).

---

### 1. What exactly is "mood" and how is it different from "style"?
Mood/atmosphere is the *global* look of a photo: colour palette, colour temperature (warm/cool),
brightness and contrast (tone curve), saturation and lighting (golden hour, blue dusk, fog, neon,
noir). Artistic *style* also includes **texture** — brush strokes, patterns. For mood transfer on
photographs we want the colour/tone/lighting but *not* the texture, otherwise the output stops
looking like a photo. That is why we add the mood layer (colour-only/tone modes + guided filter) on
top of AdaIN.

### 2. Where is the deep learning in your pipeline?
(a) **Gatys**: a pretrained VGG-19 defines content and style losses and we back-propagate to the pixels.
(b) **AdaIN**: a pretrained VGG-19 encoder → AdaIN in feature space → a pretrained CNN decoder. The
mood statistics are computed on 512-channel deep features at `relu4_1`, not on pixels. The DIP parts
(Lab, guided filter) refine the network's output.

### 3. Did you train anything?
No. All weights are pretrained and downloaded by `scripts/download_weights.py`: AdaIN decoder +
normalised VGG-19 from the public PyTorch port (naoto0804/pytorch-AdaIN, trained on MS-COCO + WikiArt
by its authors), torchvision VGG-19 (ImageNet), LPIPS, CLIP (LAION-2B), DeepLabV3. `train_adain.py`
was not needed because the pretrained decoder was available.

### 4. Why VGG?
VGG-19 features are the standard perceptual feature space for style transfer: plain stacked 3×3 convs,
no residual shortcuts, so each layer has a clean receptive-field hierarchy (low layers = colour/edges,
high layers = object parts). Gatys et al. and Huang & Belongie both use it, and the pretrained AdaIN
decoder was trained against it.

### 5. Why does AdaIN use a *different* VGG-19 than Gatys?
A decoder only inverts the exact feature space it was trained on. The released AdaIN decoder was
trained on a *normalised* VGG-19 (from the original Torch code) whose first 1×1 conv performs the
Caffe preprocessing (RGB→BGR, ×255, mean subtraction). So AdaIN takes raw RGB in [0,1] with **no**
ImageNet normalisation. Gatys only optimises pixels, so any good VGG works; we use torchvision's
ImageNet VGG-19 with ImageNet mean/std normalisation. We verified compatibility visually: at α = 0 the
decoder reproduces the content image.

### 6. What does AdaIN do mathematically?
`AdaIN(x, y) = σ(y)·(x − μ(x))/σ(x) + μ(y)`, per channel, with μ, σ computed over the spatial
dimensions. It normalises away the content's channel statistics and imposes those of the reference.
It is Reinhard's colour transfer, but applied to 512 deep feature channels instead of 3 Lab channels.
It has no learnable parameters; all the learning is in the decoder.

### 7. Why do channel means/stds capture "style"/mood?
Each VGG channel responds to a pattern (a colour, an edge orientation, a texture). Its spatial mean
says *how much* of that pattern the image has, and its std says how strongly it varies, regardless of
*where*. Throwing away the spatial layout keeps the "look" and discards the "layout". This is the same
idea as the Gram matrix (Li et al. 2017 showed matching Gram matrices ≈ matching feature statistics,
a form of distribution alignment).

### 8. Why Gram matrices in Gatys?
`G_l = F_l F_lᵀ / (C·H·W)`: entry (i, j) is the correlation of channels i and j summed over all
positions. It is spatially invariant (texture/colour statistics without layout). Matching Gram
matrices at several layers matches textures from fine (relu1_1) to coarse (relu5_1).

### 9. Content loss vs style loss trade-off?
`L = w_c L_content + w_s L_style + w_tv L_TV`. A larger `w_s/w_c` gives stronger mood/texture but
loses structure. We use w_c = 1, w_s = 1e5, w_tv = 1e-2 and initialise from the content image, so the
optimisation starts from the photo and only moves as far as the style term demands.

### 10. Why initialise Gatys from the content image and not noise?
Starting from noise needs many more iterations and gives a painterly reconstruction. Starting from the
content keeps the result near the photograph (more photographic, faster convergence). It is a common
choice for photo-realistic use.

### 11. What does α do?
In AdaIN, `t' = α·AdaIN(f_c, f_s) + (1−α)·f_c` before decoding. It interpolates in feature space
between "content reconstruction" (α = 0) and "full transfer" (α = 1). For classical methods it is a
pixel blend. Our α sweep figure shows a smooth, monotone change.

### 12. Why is α = 0 not exactly the content for AdaIN?
α = 0 gives `decoder(encoder(content))`. The decoder was trained with content and style losses, not to
reconstruct pixels exactly, and `relu4_1` is at 1/8 resolution, so fine detail is lost. Measured
reconstruction SSIM: ≈0.60 at 512 px (astronaut), lower at small sizes. In our "AdaIN + mood"
pipeline, α = 0 returns ≈ the content (SSIM > 0.99) because luminance comes from the content.

### 13. Why does plain AdaIN look painterly on photographs?
The decoder was trained on WikiArt paintings as styles, so it reproduces brush-stroke-like textures.
Matching statistics at `relu4_1` also changes mid-level texture features, and the decoder's
reconstruction distorts fine structure (faces, text, straight lines). Our metrics show it: plain AdaIN
has the lowest SSIM and highest LPIPS of all methods.

### 14. How does your photorealistic refinement fix that?
Two steps. (1) **Colour-only mode**: in Lab we keep the content's L* (all edges and texture live mostly
in luminance) and take only a*, b* from the AdaIN output. (2) **Guided filter** with the content as
guide: in every window the output is forced to be a local affine function of the content colours,
`q = aᵀI + b`, so no edge can appear where the content has none, and AdaIN textures are smoothed
away. This is the smoothing idea of PhotoWCT and the locally-affine colour model of Luan et al.

### 15. Why a guided filter and not a bilateral filter?
The bilateral filter smooths the *stylised* image using its own (already corrupted) edges and can
create gradient reversal. The guided filter uses the *clean content image* as the edge guide, runs in
O(1) per pixel (box filters), and has a closed-form local linear model. We implemented the 3-channel
colour-guide version ourselves (3×3 covariance inverse in closed form). Bilateral is still offered as
an option.

### 16. Why Lab colour space?
Lab separates lightness (L*) from chroma (a*: green–red, b*: blue–yellow), and it is approximately
perceptually uniform. That gives (a) clean "keep luminance, change colour" operations, (b) per-channel
statistics with little cross-channel correlation (Reinhard's insight), and (c) meaningful distances for
the k-means palette and our metrics. YCbCr is a cheaper linear alternative (implemented in utils);
Lab is better because it is perceptual.

### 17. Why does Reinhard (or histogram matching) fail?
They only match *global* per-channel statistics: (i) no spatial awareness, so a reference with a big
orange sky forces orange onto everything, including faces and grass; (ii) independent channels, so RGB
histogram matching can create colour casts and over-saturation; (iii) no semantics, so they can't tell
sky from foreground. They are, however, fast, structure-preserving and surprisingly strong on pure
colour statistics, which our metrics show honestly.

### 18. How does histogram matching work?
Compute the CDF of the source channel T(r) and of the reference G(z), then map every value
r → G⁻¹(T(r)). The output has (approximately) the reference histogram. We implemented it with
1024-bin CDFs and linear interpolation of the inverse CDF (no posterisation).

### 19. How do you blend multiple moods?
Weighted mixing of the target feature statistics: μ = Σ wᵢ μᵢ, σ = Σ wᵢ σᵢ at `relu4_1`. Because
AdaIN is affine in (μ, σ), this equals a weighted mix of the AdaIN outputs (Huang & Belongie, style
interpolation).

### 20. How did you evaluate? Why these metrics?
Content preservation: SSIM and LPIPS on luminance between output and content (luminance because a mood
change *should* change colour). Mood match: Lab mean/std distance, Lab Wasserstein distance and RGB
Bhattacharyya histogram distance to the reference, plus CLIP image–image similarity (and its gain over
the unmodified content) and CLIP image–text similarity with a mood prompt. Speed in s/image. The fast
methods (Reinhard, histogram, AdaIN, ours in two modes) ran on all 10 × 6 pairs; Gatys ran on a
documented 3 × 6 subset at 384 px.

### 21. Isn't there a trade-off in your metrics?
Yes, and it is the central result. Methods that change pixels more (Gatys, plain AdaIN) match the
reference's colour statistics better but destroy structure. Our pipeline sits at the photographic end:
near-identity structure (SSIM 0.996) with a clear mood shift (Lab distance halved, CLIP-text gain on par
with Reinhard); the color+tone mode moves further along the trade-off. Classical methods match *global* colour
statistics best, by construction, because that is exactly what they optimise.

### 22. Why does CLIP similarity matter if it's not a pixel metric?
CLIP embeds images by *semantic* content and atmosphere. ΔCLIP (output-to-reference similarity minus
content-to-reference similarity) measures how much the atmosphere moved toward the reference. The
text-prompt version checks the mood against a description ("a photo in warm golden hour light").

### 23. What is the mood descriptor?
A 5-colour palette from k-means in Lab (cluster shares give the swatch widths), plus mean L*
(brightness), std L* (RMS contrast), mean HSV saturation, Hasler–Süsstrunk colourfulness, the
correlated colour temperature of the mean colour via McCamy's formula (gray-world illuminant estimate),
and a rule-based label (e.g. "warm, mid-key, vivid").

### 24. What does the semantic (stretch) extension do?
DeepLabV3-MobileNetV3 (pretrained, PASCAL-VOC classes) predicts a foreground-object probability. We
refine it with a guided filter so it snaps to edges, then blend: background gets the full mood and
foreground objects (people, vehicles, animals) get a weaker or different mood. Limitation: VOC has no
"sky" class, so it is foreground/background, not sky/ground.

### 25. Main limitations?
(1) With colour-only mode, moods that are mainly about *brightness* (noir, night) are only partly
transferred, which is why the tone and colour+tone modes exist. (2) Global statistics can't relight a
scene: no new light sources or shadows (a midday photo can't get real sunset shadows). (3) Mismatched
scene content (e.g. a neon street reference applied to a landscape) gives implausible colour
placement. (4) AdaIN decoder resolution: fine detail is lost at α = 1 without refinement. (5) Metrics
are proxies; a user study would be the real test (template in the report appendix).

### 26. Ethical considerations?
Mood transfer is image manipulation. It can change how real events are perceived (making a scene look
gloomier or more dramatic). Edited photos used in journalism or evidence should be disclosed. We only
used freely licensed images and credit every author in `data/*/CREDITS.md`.

### 27. How would you improve it?
Semantic correspondence (match sky to sky, as in Luan et al.'s segmentation-guided transfer), WCT or
learned photorealistic transfer (PhotoWCT2, WCT²), local tone mapping/relighting, video temporal
consistency, and a proper user study.

---

## Numbers to remember (from `results/metrics_summary.md`, `results/speed.csv`)

| | SSIM ↑ | LPIPS ↓ | Lab dist ↓ | ΔCLIP text ↑ | s/img (MPS) |
|---|---|---|---|---|---|
| Content unchanged | 1.000 | 0.000 | 50.1 | 0 | – |
| Reinhard | 0.809 | 0.114 | 6.9 | 0.089 | 0.010 |
| Histogram (Lab) | 0.706 | 0.270 | 8.2 | 0.083 | 0.048 |
| AdaIN | 0.331 | 0.477 | 13.3 | 0.096 | 0.12 |
| **Ours, color mode** | **0.996** | **0.006** | 26.8 | 0.086 | 0.14 |
| **Ours, color+tone** | 0.757 (0.86 w/o noir) | 0.192 | 16.8 | **0.108** | 0.12 |
| Gatys (18-pair subset, 384 px, 300 it.) | 0.638 | 0.265 | 21.6 | 0.072 | 22.2 |

* Dataset: 10 contents × 6 moods = 60 pairs; Gatys on 3 × 6 = 18 pairs.
* Refinement example (vineyard × neon): SSIM raw AdaIN 0.22 → colour mode 0.98 → + guided filter 0.994.
* Speed: ours 0.10 s (MPS) / 0.39 s (CPU) vs Gatys 20.6 s / 57.9 s, about 200× faster.
* Guided filter: r = long side/16 (32 px at 512), ε = 1e-2, ~27 ms at 512 px; halves the high-frequency chroma noise (4.41 → 1.96).
* AdaIN α = 0 reconstruction SSIM ≈ 0.60 at 512 px.
