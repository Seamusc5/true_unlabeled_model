# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A student research project: can a CNN learn **chirality** (S vs. mirror-S) from *unlabeled* data? Labeled training works; the open problem is the unlabeled (self-supervised) objective. Two scripts, no package, no test suite, no linter config.

## Commands

```bash
python train_unlabeled_cnn.py     # generates data in memory, trains 12 epochs, saves model
python test_unlabeled_model.py    # loads model, evaluates on 500 S + 500 mirror-S (seed 43)
MPLBACKEND=Agg python ...          # headless: both scripts call plt.show(), which otherwise blocks
```

Requires TensorFlow ≥ 2.16 (Keras 3), NumPy, Pillow, Matplotlib, scikit-learn (eval only).

Side effects to be aware of:
- The model is written to `../s_anti_s_blind_tanh.keras` — the **parent** of the repo (`RESEARCH_DIR`), and the training script **deletes any existing file there first**.
- `train_sample_preview.png` goes to the current working directory; eval PNGs go next to `test_unlabeled_model.py`. `results/` holds committed copies from an earlier run.
- Glyphs come from the hard-coded font `/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf`; if it is missing, Pillow's default font is used **silently** (different shape, no size variation).

## Architecture

`test_unlabeled_model.py` imports `generate_synthetic_data`, the two custom layers and `RESEARCH_DIR` from `train_unlabeled_cnn.py` — changing those signatures/names breaks evaluation, and the custom layers must stay registered (`register_keras_serializable`) for `load_model` to work.

Data (`generate_synthetic_data`): italic "S" drawn at (10,10) on a 35×35 canvas, random font size 10–22 and rotation 0–359°, resized to 36×36, inverted to bright-on-dark. Each S is paired with its horizontal mirror. Returns `X` (N,36,36,1) and `y` (+1 = S, −1 = mirror-S); the trainer discards `y`.

Model (`build_antisymmetric_cnn`): `CenterOfMassAlignLayer` (integer `tf.roll` to center intensity) → a shared `feature_scorer` CNN `g` applied to the image and to its horizontal flip → `tanh(g(x) − g(flip x))`. The output is therefore **odd under horizontal flip by construction** (exact up to center-of-mass rounding).

Training (`BlindChiralTrainer`): custom `train_step` on unlabeled batches. It mirrors each batch about a **random axis** (`RandomRotation(0.5)` applied after `flip_left_right`) and minimizes `−mean(((P − P_mirror)/2)²)`.

Evaluation: thresholds the output at 0 and, because an unlabeled model can only be right up to a global sign, flips the sign if accuracy < 50 %. Accuracy near 50 % (or outputs ≈ 0) means nothing was learned.

## Unlabeled objective: what fails and why (details in `docs/unlabeled_chirality_investigation.md`)

- Don't revert to a fixed horizontal flip in the loss. Changing sign under one fixed reflection is also satisfied by "does the shape lean / or \". The network learns that tilt instead: |W| → 1 but accuracy ≈ 50 %, with the output sign flipping every 90° of rotation. It even reaches |W| ≈ 1 on an achiral letter "I". Chirality must change sign under **every** reflection, hence the random-axis mirror.
- The loss must be *maximized* in |P − P_mirror|. The original code minimized `square(abs(P − P_mirror)/2.0 * 0.5)`, which equals `P²/4`, so the output collapsed to 0.
- Because the model already changes sign under the flip, the random-axis loss equals a magnitude term plus a rotation-consistency term `−f(x)·f(Rx)`. The consistency term alone collapses to f ≈ 0 on about half of seeds, so keep both.

## Data caveats (`generate_synthetic_data`)

- No random translation: the glyph is always drawn at (10,10) and rotated about the canvas centre, so its position lies on a circle (radius set by font size) locked to the rotation angle. `CenterOfMassAlignLayer` is what removes this position/orientation cue; don't drop it without adding real random shifts.
- `randint` is inclusive: font sizes 10–23, angles 0–360 (0 and 360 identical). Sizes 22–23 are often clipped at the image border.
- Only 14 × 361 possible images per class: 5000 training S contain ~3000 distinct images, and ~65 % of the test S images (seed 43) are pixel-identical to training images, so the test set is not truly held out.
- Every mirror-S is the exact pixel mirror of an S in the same set.
