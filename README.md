# Lightweight Test-Time Image Recovery for Robust Medical Image Classification

Recovering a **frozen** medical image classifier from corruption-induced
collapse using a **1,119-parameter** autoencoder placed in front of it — no
retraining, no access to the classifier's weights.

> **Status: work in progress.** The pipeline runs end to end and the core result
> reproduces, but the reported numbers come from a single unseeded run and are
> not yet publication-ready. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
> for exactly what is done and what is outstanding.

---

## The problem

A pneumonia classifier trained on clean X-rays does not degrade gracefully under
image corruption — it **collapses to a single class**. And it does so *silently*:
because the dataset is imbalanced, raw accuracy stays near the majority-class
rate while the model has stopped discriminating entirely.

| Metric | Clean | Gaussian noise (severity 3/5) |
|---|---|---|
| Balanced accuracy | 0.830 | **0.500** ← chance |
| Raw accuracy | high | still near majority-class rate |

Any evaluation reporting raw accuracy alone would miss this completely. Balanced
accuracy is the headline metric throughout this work for that reason.

## The approach

Most robustness work fixes the classifier — augmentation, fine-tuning,
robust training. All of it requires *retraining*. This project asks a different
question: **what if the classifier cannot be touched?** Regulatory-frozen
deployments, third-party models, on-device inference.

So the classifier is frozen after training and a small recovery module is
inserted upstream:

```mermaid
flowchart LR
    A["corrupted X-ray"] --> B["Recovery AE<br/>1,119 params<br/><b>trained</b>"]
    B --> C["MobileNetV2<br/>2.2M params<br/><b>FROZEN</b>"]
    C --> D["prediction"]
```

The module is trained with a perceptual loss guided by the frozen classifier:

```
loss = MSE(recon, clean)  +  λ · CE(frozen_clf(recon), label)
```

λ ramps up from 0 after a warmup. **The CE term is the mechanism, not a
refinement** — with MSE alone the autoencoder converges to the identity map and
recovers nothing at all.

Freezing matters methodologically: because the classifier never sees a
corruption, any accuracy recovered is attributable to the recovery module rather
than to the classifier quietly learning the corruption distribution.

## Preliminary results

> Single unseeded run, 64×64, three corruption families. **Treat as directional.**
> Error bars and the full corruption set are the top outstanding items.

Balanced accuracy on PneumoniaMNIST test:

| Method | Retrains classifier? | Clean | Noise (sev 3/5) |
|---|---|---|---|
| Baseline 1 — clean-trained, frozen | — | 0.830 | 0.500 |
| 3×3 box denoiser (0 params) | no | — | ~0.50–0.65 |
| **Recovery AE (ours, 1,119 params)** | **no** | ~0.74 | **0.732** |
| Baseline 2 — MedMNIST-C augmentation | **yes** | ~0.90 | ~0.78 |

Four findings worth stating plainly:

1. **A ~1k-parameter module lifts a frozen classifier off chance** (0.500 → 0.732),
   including at a severity held out of recovery training.
2. **Learning is doing the work.** The zero-parameter box denoiser recovers far
   less, so this is not a claim about smoothing.
3. **Capacity does not help.** Widths 4/8/16/32 were swept; w=4 gives the best
   mean gain. Lightweight recovery appears not merely feasible but *preferable*.
   ⚠️ This is the load-bearing claim and it currently has no error bars.
4. **It generalizes off-distribution.** Positive gain on unseen *compound*
   corruptions (+0.16 to +0.24) and on Kermany chest X-rays — a different
   dataset entirely.

**We do not beat augmentation, and the paper should not claim to.** Baseline 2
is stronger — but it retrains the classifier, which is precisely what this
setting rules out. The contribution is a *different mechanism*, not a better
score. See [`docs/FINDINGS.md`](docs/FINDINGS.md) for the full framing.

## Honest limitations

- **Recovery is classifier-coupled.** An AE trained against classifier A does
  not transfer to classifier B — B collapses on A's recovered images. The module
  learns something tuned to its guidance classifier's decision surface, not a
  universal restoration.
- **On clean input the AE slightly *hurts* accuracy.** It "fixes" images that
  aren't broken. This motivates a corruption-gating detector (future work).
- **64×64, not 224×224.** Absolute accuracy is therefore not comparable to the
  MedMNIST-C paper's tables. The relative findings are the contribution.
- **Three corruption families, three severities.** The registry has more. The
  selection needs widening before it can be called comprehensive.

## Setup

```bash
git clone https://github.com/FahimSadikRashad/medical-lightweight-recovery-pipeline.git
cd medical-lightweight-recovery-pipeline

bash scripts/install.sh          # ImageMagick + Python deps, in the required order
python scripts/download_data.py  # PneumoniaMNIST (add --kermany for the transfer set)
```

`medmnistc` renders corruptions through ImageMagick via `wand`, so the system
library must be installed *before* `wand` and `wand` reinstalled after. That
ordering is why installation is a script rather than just `requirements.txt`.

<details>
<summary><b>Running on Colab</b></summary>

Put outputs on Drive so a session timeout never costs a run:

```python
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ["ROBUSTMED_ROOT"] = "/content/drive/MyDrive/cse6207"

!bash scripts/install.sh
!python scripts/01_baseline1_frozen.py
```

Set `ROBUSTMED_ROOT` **before** importing `robustmed` — `config` resolves paths
at import time. Use a GPU runtime (Runtime → Change runtime type → GPU).
</details>

## Reproducing

Stages are independent. Each writes JSON to `$ROBUSTMED_ROOT/results/`, and
later stages read from there rather than depending on earlier in-memory state —
so you can rerun any single stage without repeating the ones before it.

| Stage | Produces |
|---|---|
| `01_baseline1_frozen.py` | clean-trained classifier + the collapse table |
| `02_baseline2_augmented.py` | MedMNIST-C augmentation baseline (Paper A) |
| `03_train_recovery.py` | recovery AEs at every width, + params/latency |
| `04_evaluate_recovery.py` | main RQ: gain vs capacity, incl. held-out severity |
| `05_ablation.py` | vs box denoiser; recovery stacked with augmentation |
| `06_generalization.py` | compound corruptions + Kermany transfer |
| `07_figures.py` | all figures, regenerated from `results/` |

```bash
python scripts/01_baseline1_frozen.py
# ... through 07
```

`--limit 500` gives a fast end-to-end smoke run; `--epochs N` shortens training.
Training stages resume from checkpoint automatically, so an interrupted
overnight run can just be restarted.

`07_figures.py` skips figures whose experiment hasn't run yet, so it is safe to
call at any point while work is still in progress.

## Layout

```
robustmed/
  config.py       all knobs — edit here, not in the scripts
  corruptions.py  wrapper over the official MedMNIST-C registry
  data.py         datasets and loaders (all yield [0,1] tensors)
  models.py       Classifier, ConvAE, BoxDenoiser
  engine.py       train / evaluate / latency / checkpoints
  figures.py      one function per figure, reads from results/
  store.py        JSON result persistence
scripts/          one file per stage, plus install & download
docs/
  EXPERIMENTS.md  what's done, what's left ← start here when resuming
  FINDINGS.md     results and interpretation, including what failed
legacy/           the original single-file Colab export, unmodified
```

## Two conventions worth knowing

**Severity is 0-indexed in code.** `medmnistc`'s `apply(img, severity)` indexes
5 levels from 0; the MedMNIST-C paper numbers them 1–5. Code is 0-indexed
throughout and `data.pretty()` adds 1 for figure labels only. This README uses
the paper's 1-indexed form.

**Images are `[0,1]` and unnormalized.** Every dataset yields float CHW tensors
in `[0,1]`; ImageNet normalization lives inside `models.Classifier`. That is
what lets a single loader feed both the autoencoder (pixel space) and the
classifier.

## Adding an experiment

1. Write `scripts/NN_name.py`, loading what it needs via `store.load(...)`.
2. Add a name constant to `robustmed/store.py`.
3. `store.save(NAME, payload)` at the end.
4. For a figure: add a function to `robustmed/figures.py` and call it from
   `07_figures.py` behind `store.load_optional(...)`, so the figure script keeps
   working before the experiment has been run.
5. Record it in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Data & prior work

- **PneumoniaMNIST** — Yang et al., *MedMNIST v2*, Scientific Data 2023.
- **MedMNIST-C** — Di Salvo et al., 2024. Corruptions come from the authors'
  official registry, not hand-rolled noise, so the fault model is the
  benchmark's rather than ours.
- **Kermany chest X-ray** — Kermany et al., Cell 2018. Cross-domain transfer set.
- **Test-time recovery lineage** — Gao et al., CVPR 2023; *Decorruptor*, ECCV
  2024. This work instantiates that idea at ~1k parameters instead of a
  diffusion model.

## Citation

Not yet published. Please cite the repository in the meantime:

```bibtex
@misc{rashad_lightweight_recovery,
  author = {Rashad, Fahim Sadik},
  title  = {Lightweight Test-Time Image Recovery for Robust Medical Image Classification},
  year   = {2026},
  url    = {https://github.com/FahimSadikRashad/medical-lightweight-recovery-pipeline}
}
```
