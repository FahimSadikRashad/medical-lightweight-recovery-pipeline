# Colab runbook — what to run, in what order, with which config

Operational companion to [`PRESENTATION_FEEDBACK.md`](PRESENTATION_FEEDBACK.md)
(what needs doing and why) and [`EXPERIMENTS.md`](EXPERIMENTS.md) (the ledger).
This file is only *how to execute it*.

---

## The golden rule: one `ROBUSTMED_ROOT` per experimental configuration

`config.ROOT` is resolved **at import time**, and `store.py` binds `RESULT_DIR`
at import too. So the output root cannot be changed from inside a running
script — but that is exactly what makes it a clean isolation mechanism.

**Every time you change dataset, seed, or a training hyperparameter, change the
root.** Result filenames (`baseline1_frozen.json`, `recovery_ae_w4.pt`, …) do not
encode the config, so two configs sharing a root will silently overwrite each
other.

```
/content/drive/MyDrive/cse6207/
├── pneumonia_main/         Phase A + B  (headline run)
├── pneumonia_mse_only/     Phase D      (AE_LAMBDA_MAX = 0)
├── pneumonia_seed1/        Phase F
├── pneumonia_seed2/
├── blood_main/             Phase E
└── derma_main/
```

This costs nothing and makes every run independently re-readable. It is the one
convention in this file you must not skip.

---

## Cell 0 — session setup (run first, every session)

```python
# --- mount Drive and pick the run root -------------------------------------
from google.colab import drive
drive.mount('/content/drive')

import os
RUN = "pneumonia_main"          # <<< CHANGE THIS PER PHASE
os.environ["ROBUSTMED_ROOT"] = f"/content/drive/MyDrive/cse6207/{RUN}"

# --- get the code ----------------------------------------------------------
!git clone -q https://github.com/FahimSadikRashad/medical-lightweight-recovery-pipeline.git 2>/dev/null || true
%cd /content/medical-lightweight-recovery-pipeline
!git pull -q                     # picks up the to_tensor01 warning fix

# --- install (~3 min; ImageMagick before wand, order matters) --------------
!bash scripts/install.sh

print("ROOT =", os.environ["ROBUSTMED_ROOT"])
```

Confirm GPU first: **Runtime → Change runtime type → GPU**. `install.sh` prints
the device and will say `CPU ONLY` if you forgot.

### Config-patching helper (keep this cell around)

Scripts run as **subprocesses** (`!python scripts/...`), so patching
`config.<X>` in the notebook has no effect — the edit must go into the file.

```python
import re, pathlib

def patch_config(**kv):
    """Rewrite top-level assignments in robustmed/config.py."""
    p = pathlib.Path("robustmed/config.py"); s = p.read_text()
    for k, v in kv.items():
        s, n = re.subn(rf'^{k} = .*$', f'{k} = {v!r}', s, flags=re.M)
        assert n == 1, f"{k}: matched {n} lines, expected 1"
        print(f"  {k} = {v!r}")
    p.write_text(s)

def show_config(*keys):
    s = pathlib.Path("robustmed/config.py").read_text()
    for k in keys:
        print(re.search(rf'^{k} = .*$', s, flags=re.M).group(0))
```

> `patch_config` asserts it matched exactly one line, so a typo fails loudly
> instead of silently doing nothing.

---

## Phase A — lock the headline PneumoniaMNIST run

**Config:** stock defaults. `RUN = "pneumonia_main"`.

Stages 01 and 02 are already done, but their outputs went to an ephemeral
`runs/`. Redo them under the Drive root so everything downstream is reproducible
from one place.

```python
!python scripts/download_data.py
!python scripts/01_baseline1_frozen.py      # ~8 min
!python scripts/02_baseline2_augmented.py   # ~25 min (augmentation is CPU-bound)
!python scripts/03_train_recovery.py        # ~45-90 min, the long one
!python scripts/04_evaluate_recovery.py     # ~10 min
!python scripts/05_ablation.py              # ~15 min
!python scripts/07_figures.py               # ~2 min
```

**Expected, from the run you already did:** B1 clean `0.790`, B1 collapses to
`0.500` on noise at all severities, B2 clean `0.880` and no collapse anywhere.
If B1's clean number differs by more than ~0.02, stop and check the seed.

**What to look at afterwards:**

```python
import pandas as pd, os
R = os.environ["ROBUSTMED_ROOT"] + "/results"
print(pd.read_csv(f"{R}/recovery_sweep_summary.csv").to_string(index=False))
print(pd.read_csv(f"{R}/ablation.csv").to_string(index=False))
```

**Decision gate.** `recovery_sweep_summary.csv` has a `mean_gain` column per
width. Whichever width wins is your headline model. Stage 04 prints a warning if
it disagrees with `HEADLINE_WIDTH = 4`. If it does disagree:

```python
patch_config(HEADLINE_WIDTH=8)   # or whatever won
```
then re-run stages 05 and 07. **Do not skip this** — slide 7's entire claim
("the smallest module wins") lives in that column.

---

## Phase B — full corruption grid (remark 3, first half)

**Config:** all 13 corruptions × all 5 severities. Same root as Phase A — this
is evaluation only, it adds result files rather than overwriting checkpoints.

```python
from robustmed import corruptions
names = corruptions.names()
print(len(names), "corruptions:", names)

patch_config(EVAL_CORRUPTIONS=names, EVAL_SEVERITIES=[0, 1, 2, 3, 4])
show_config("EVAL_CORRUPTIONS", "EVAL_SEVERITIES")
```

Then re-run **evaluation only** — no retraining needed, the checkpoints already
exist and every stage resumes from them:

```python
!python scripts/01_baseline1_frozen.py      # resumes from ckpt, re-evaluates
!python scripts/02_baseline2_augmented.py   # resumes from ckpt, re-evaluates
!python scripts/04_evaluate_recovery.py
!python scripts/05_ablation.py
```

**Runtime warning.** Corruptions are rendered by ImageMagick on CPU, one image
at a time, and free Colab gives you ~2 vCPUs. 66 conditions × 624 images × 4
widths is ~165k corruption renders. Budget **2–4 hours**, and expect the GPU to
sit idle — this is CPU-bound, not GPU-bound.

If that is too slow, the fix is a code change worth requesting: cache the
corrupted test tensors once to `.npz` and reuse them across all models, instead
of re-rendering per model.

**What to look for.** Group results by category. The six photometric
corruptions (`brightness_up/down`, `contrast_up/down`, `gamma_corr_up/down`) are
the ones to watch — they are near-invertible intensity shifts rather than
information-destroying degradations, and the AE was trained only on the latter.
A near-zero or negative gain there is a *finding*, not a failure. See
`PRESENTATION_FEEDBACK.md` §3.

> **Known limitation:** `figures.fig_collapse_recovery` sizes itself as
> `1.05 × n_conditions` inches, so 66 conditions produces a ~72-inch-wide
> unusable image. Either keep `07_figures.py` on the reduced 3-corruption set, or
> ask for a grouped-by-category figure — that needs a code change.

---

## Phase C — efficiency numbers (remark 2)

Stage 03 already recorded params and GPU latency into `recovery_cost.json`. Two
additions, both cheap.

### C1 — cost as a fraction of the classifier (the free anchor)

```python
import json, os
from robustmed import models
cost = json.load(open(os.environ["ROBUSTMED_ROOT"] + "/results/recovery_cost.json"))
clf_params = models.count_params(models.Classifier(2))
print(f"classifier: {clf_params:,} params")
for w, c in cost.items():
    pct = 100 * c["params"] / clf_params
    print(f"  AE w={w:>2}: {c['params']:>7,} params = {pct:.3f}% of classifier")
```

This is the number that makes "lightweight" mean something — right now you only
compare the module against other sizes of itself.

### C2 — CPU single-thread latency (your title says "Constrained Hardware")

Every timing you have is from a T4. Force CPU by hiding the GPU **before** any
import — `engine.DEVICE` is resolved at import time:

```python
# fresh runtime, or this will not take effect
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ROBUSTMED_ROOT"] = "/content/drive/MyDrive/cse6207/pneumonia_main"

import torch; torch.set_num_threads(1)
from robustmed import config, engine, models
print("device:", engine.DEVICE)          # must print cpu

for w in config.AE_WIDTHS:
    ae = engine.load_recovery(w)
    print(f"w={w:>2}  {engine.latency_ms(ae, batch_size=1, iters=50):.3f} ms/img (cpu, 1 thread)")

clf = models.Classifier(2).to(engine.DEVICE)
print(f"classifier {engine.latency_ms(clf, batch_size=1, iters=50):.3f} ms/img — report the ratio")
```

**Still needs a code change:** MACs/FLOPs (via `thop`/`fvcore`), peak memory, and
a heavy recovery baseline for the upper cost anchor. Ask when you want these.

---

## Phase D — MSE-only ablation (nearly free, currently missing from every table)

You know MSE-only training fails; it is your cleanest evidence that the
perceptual loss *is* the mechanism. It is in no table.

**New root** — this retrains the AEs and would otherwise overwrite Phase A's
checkpoints, since `ae_ckpt(width)` does not encode lambda.

```python
os.environ["ROBUSTMED_ROOT"] = "/content/drive/MyDrive/cse6207/pneumonia_mse_only"
```

Restart the runtime, re-run Cell 0 with `RUN = "pneumonia_mse_only"`, then:

```python
patch_config(AE_LAMBDA_MAX=0.0, EVAL_CORRUPTIONS=["gaussian_noise","gaussian_blur","jpeg_compression"], EVAL_SEVERITIES=[0,2,4])
!python scripts/01_baseline1_frozen.py     # need B1 in this root as the frozen guide
!python scripts/03_train_recovery.py
!python scripts/04_evaluate_recovery.py
```

**Expected:** gains at or near zero — the AE converges to the identity map.
Compare `recovery_sweep_summary.csv` across the two roots.

---

## Phase E — full suite on a new dataset (remark 1)

`bloodmnist` is the primary pick: 8 classes, roughly balanced, 3,421 test
images, natively RGB. It is the control that tests whether your collapse finding
depends on class imbalance. (`breastmnist` has only **156** test images — too
few to carry a claim across many conditions.)

**New root, new runtime.** Re-run Cell 0 with `RUN = "blood_main"`, then:

```python
patch_config(DATA_FLAG="bloodmnist")

# each dataset has its OWN corruption set - read it, don't assume
from robustmed import corruptions
names = corruptions.names()
print(len(names), names)
patch_config(EVAL_CORRUPTIONS=names, EVAL_SEVERITIES=[0, 1, 2, 3, 4],
             TRAIN_CORRUPTIONS=names[:3])   # or a category-spanning subset

!python scripts/download_data.py
!python scripts/01_baseline1_frozen.py
!python scripts/02_baseline2_augmented.py
!python scripts/03_train_recovery.py
!python scripts/04_evaluate_recovery.py
!python scripts/05_ablation.py
!python scripts/07_figures.py
```

**Three things change with 8 classes — read results accordingly:**

1. **Chance balanced accuracy is 0.125, not 0.50.** A "collapse" is a much
   larger drop. `figures` still draws its reference line at 0.5 — that line is
   wrong for this dataset and needs a code change to parameterize.
2. `engine.collapsed()` counts distinct predictions, so it generalizes, but with
   8 classes partial collapse (say 8 → 2 classes) will not trip it. Look at
   `pred_counts` in the JSON directly.
3. **Re-quote the parameter count.** BloodMNIST is natively 3-channel like
   PneumoniaMNIST-as-RGB, so w=4 should still be 1,119 — but verify with
   `models.count_params(models.ConvAE(4))` rather than reusing the number.

**The result that matters:** if collapse reproduces on a balanced dataset, your
mechanism is a genuine representation failure. If it does not, part of your
finding is an imbalance artifact — a real scope limit, better found by you.

Then repeat with `RUN = "derma_main"`, `DATA_FLAG="dermamnist"` (7 classes,
heavily imbalanced, delivers the dermatology promise from slide 9).

---

## Phase F — multi-seed (the top statistical gap)

"The smallest module wins" has no error bars, and it is slide 7's whole claim.
Three seeds minimum.

```python
# one root per seed, one runtime each
for tag, seed in [("pneumonia_seed1", 1), ("pneumonia_seed2", 2)]:
    ...  # Cell 0 with RUN = tag, then:
    #   patch_config(SEED=seed)
    #   run stages 01 -> 04
```

Then aggregate `mean_gain` per width across `pneumonia_main`,
`pneumonia_seed1`, `pneumonia_seed2` and report mean ± std. **If the spread
between widths is smaller than the spread across seeds, "smallest wins" is not
supported** and slide 7 needs rewording. Better to know now.

---

## Phase G — blocked on code changes

Not runnable as-is. Ask and I will implement.

| Item | Needs |
|---|---|
| MACs/FLOPs + peak memory | `thop`/`fvcore` in `engine`, extend `RECOVERY_COST` |
| Grouped-by-category figure | `figures.fig_collapse_recovery` rewrite for 66 conditions |
| Configurable chance line | `figures` currently hardcodes `0.5` |
| Corrupted-tensor caching | big Phase B speedup; `.npz` cache keyed by (dataset, corruption, severity) |
| `chestmnist` pneumonia adapter | multi-label → binary slice of label 6 |
| RSNA external transfer | DICOM loader, own splits, borrowed corruption registry |
| Lightweight baselines (DnCNN/NAFNet) | new models + training path |
| Heavy diffusion baseline | cost anchor for the "lightweight" claim |

---

## Suggested order, with rough time

| # | Phase | Time | Why here |
|---|---|---|---|
| 1 | A — headline run on Drive | ~2.5 h | Everything else reads from it |
| 2 | C1 — cost as % of classifier | 5 min | Free, answers remark 2 immediately |
| 3 | D — MSE-only | ~1 h | Converts a known result into evidence |
| 4 | B — full corruption grid | 2–4 h | Answers remark 3; surfaces photometric risk |
| 5 | C2 — CPU latency | 15 min | Needs a CPU runtime, so batch it separately |
| 6 | E — bloodmnist full suite | ~3 h | Answers remark 1 properly |
| 7 | F — multi-seed | ~6 h | Highest statistical value, most wall-clock |
| 8 | E' — dermamnist | ~3 h | Delivers the slide-9 modality promise |

Phases A–E fit in roughly two working sessions. F is best left running overnight
— all training stages resume from checkpoint, so a disconnect costs one epoch,
not the run.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `saved -> runs/...` | `ROBUSTMED_ROOT` unset or set after import | Set it in Cell 0, before any `robustmed` import; restart runtime |
| `wand`/ImageMagick import error | wand installed before the system library | Re-run `scripts/install.sh`; order matters |
| `CPU ONLY` in install output | GPU runtime not selected | Runtime → Change runtime type → GPU |
| `run scripts/01_... first` | no `baseline1_frozen.pt` in *this* root | Each root needs its own B1; it is the frozen guide |
| `patch_config` assertion fails | key not a single top-level line | `show_config(key)` and edit `config.py` by hand |
| Non-writable NumPy warnings | pre-fix `data.py` | `git pull` |
| `engine.DEVICE` is cuda when you wanted cpu | imported before setting `CUDA_VISIBLE_DEVICES` | Restart runtime; set env var first |
| Numbers differ from a previous run | different seed, or a shared root got overwritten | One root per config — check the ledger below |

### Keep a run ledger

Nothing records which config produced which root. Drop a note in each one:

```python
import json, os
json.dump({"run": RUN, "dataset": "pneumoniamnist", "seed": 0,
           "lambda_max": 1.5, "note": "headline run, full corruption grid"},
          open(os.environ["ROBUSTMED_ROOT"] + "/RUN_INFO.json", "w"), indent=2)
```
