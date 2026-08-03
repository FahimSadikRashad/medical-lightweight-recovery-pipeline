# Lightweight Test-Time Image Recovery for Robust Medical Image Classification

A ~1k-parameter convolutional autoencoder placed in front of a **frozen**
classifier, recovering accuracy under MedMNIST-C corruptions without retraining
the classifier.

- **Dataset:** PneumoniaMNIST at 64×64 (MedMNIST's native multi-resolution variant)
- **Corruptions:** the official MedMNIST-C registry (Di Salvo et al., 2024), not hand-rolled noise
- **Classifier:** ImageNet-pretrained MobileNetV2, frozen after Baseline 1
- **Metric:** balanced accuracy — the dataset is imbalanced, and raw accuracy hides the collapse
- **Transfer set:** Kermany chest X-rays

## Install

```bash
bash scripts/install.sh          # ImageMagick + python deps, in the required order
python scripts/download_data.py  # PneumoniaMNIST; add --kermany for the transfer set
```

`medmnistc` renders corruptions through ImageMagick via `wand`, so the system
library must be installed *before* `wand`. That ordering is why installation is
a script and not just `requirements.txt`.

On **Colab**, put outputs on Drive so a timeout never costs a run:

```python
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ["ROBUSTMED_ROOT"] = "/content/drive/MyDrive/cse6207"
!bash scripts/install.sh
!python scripts/01_baseline1_frozen.py
```

Set `ROBUSTMED_ROOT` **before** importing `robustmed` — `config` resolves paths
at import time.

## Run

Stages are independent. Each writes to `$ROBUSTMED_ROOT/results/`, and later
stages read from there rather than depending on earlier in-memory state.

```bash
python scripts/01_baseline1_frozen.py      # train clean-only classifier, measure collapse
python scripts/02_baseline2_augmented.py   # Paper A: MedMNIST-C augmentation baseline
python scripts/03_train_recovery.py        # capacity sweep of recovery AEs + cost
python scripts/04_evaluate_recovery.py     # main RQ table: gain vs capacity
python scripts/05_ablation.py              # vs box denoiser, vs/with augmentation
python scripts/06_generalization.py        # compound corruptions + Kermany transfer
python scripts/07_figures.py               # regenerate all figures from results/
```

Add `--limit 500` to any stage for a fast end-to-end smoke run, and `--epochs N`
to shorten training.

Training stages resume from their checkpoint automatically, so an interrupted
overnight run can just be restarted.

## Layout

```
robustmed/
  config.py       all knobs — edit here, not in the scripts
  corruptions.py  wrapper over the official MedMNIST-C registry
  data.py         datasets and loaders (all yield [0,1] tensors)
  models.py       Classifier, ConvAE, BoxDenoiser
  engine.py       train / evaluate / latency / checkpoints
  figures.py      one function per paper figure, reads from results/
  store.py        JSON result persistence
scripts/          one file per experiment stage
docs/
  EXPERIMENTS.md  what's done, what's left  <- start here when resuming
  FINDINGS.md     results and their interpretation, including what failed
legacy/           the original single-file Colab export, unmodified
```

## Two conventions worth knowing

**Severity is 0-indexed.** `medmnistc`'s `apply(img, severity)` indexes into 5
severity levels starting at 0. The MedMNIST-C paper numbers them 1–5. Code uses
0-indexed throughout; `data.pretty()` adds 1 for figure labels only.

**Images are `[0,1]`, unnormalized.** Every dataset yields float CHW tensors in
`[0,1]`. ImageNet normalization lives inside `models.Classifier`. This is what
lets one loader feed both the autoencoder (which works in pixel space) and the
classifier.

## Adding an experiment

1. Add a `scripts/NN_name.py` that loads what it needs via `store.load(...)`.
2. Add a name constant to `robustmed/store.py`.
3. `store.save(NAME, payload)` at the end.
4. If it has a figure, add a function to `robustmed/figures.py` and call it from
   `07_figures.py` behind `store.load_optional(...)` so the figure script keeps
   working before the experiment has run.
5. Record it in `docs/EXPERIMENTS.md`.

## Resolution caveat for the write-up

We work at 64×64; Paper A reports 224×224. Absolute accuracy is therefore not
directly comparable to their tables. The relative findings — collapse magnitude,
recovery gain, the capacity/gain curve — are the contribution and hold at either
resolution. State this explicitly in the paper.
