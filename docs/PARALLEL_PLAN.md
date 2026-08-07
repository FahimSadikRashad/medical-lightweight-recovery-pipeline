# Parallel execution plan — branch per experiment, across Colab + Kaggle

Companion to [`RUNBOOK.md`](RUNBOOK.md) (how to run one thing) and
[`PRESENTATION_FEEDBACK.md`](PRESENTATION_FEEDBACK.md) (what needs doing).
This file is about running many things **at once** across several accounts,
without runs overwriting each other or losing track of which config produced
which number.

---

## The model: one branch per experiment

An experiment in this project **is** a config. So make the config a commit.

```
main                    stable code. No experiment configs, ever.
├── feat/<thing>         code changes → merged into main
└── exp/<run-name>       diff from main is robustmed/config.py, nothing else
```

This replaces the `RUN_INFO.json` suggestion in `RUNBOOK.md`. The git commit is
a better provenance record than a hand-written note: it cannot drift from the
code that actually ran, and `git diff main..exp/blood-main` shows you the exact
configuration in one command.

Each runner does:

```bash
git clone -b exp/blood-main https://github.com/FahimSadikRashad/medical-lightweight-recovery-pipeline.git
```

and nothing else needs coordinating. No `patch_config` calls, no remembering
which root to set — the branch carries it.

### Results come back through git, and never conflict

Each experiment writes its result JSON/CSV into a **tracked, run-scoped**
directory and commits it:

```
results/
├── pneumonia-main/      baseline1_frozen.json, recovery_sweep.json, *.csv
├── blood-main/
├── seed1/
└── ...
```

Because every branch writes a *different* subdirectory, result commits can never
conflict. Checkpoints (`*.pt`) stay gitignored — they are large and regenerable.

**Do not merge `exp/*` into `main`** — `config.py` would conflict every time.
Pull results across surgically instead:

```bash
git checkout main
git checkout exp/blood-main -- results/blood-main/
git commit -m "results: bloodmnist full grid"
```

One command per run, no conflicts by construction.

---

## Wave 0 — code first, serially (no GPU needed)

These gate the parallel work: if an `exp/*` branch is cut before they land, it
needs a rebase later. All are small and can be done locally without a GPU.

| Branch | Change | Why it blocks |
|---|---|---|
| `feat/results-export` | tracked `results/<run>/` dir, `scripts/export_results.py`, `.gitignore` update | the aggregation mechanism above does not exist yet |
| `feat/corruption-cache` | `.npz` cache keyed by (dataset, corruption, severity) | full 13×5 grid is CPU-bound at 2–4 h per run; caching makes every Wave-1 run substantially cheaper |
| `feat/figure-fixes` | parameterize the hardcoded `0.5` chance line; group-by-category figure for 66 conditions | bloodmnist figures are *wrong* without it (chance is 0.125), and 66 conditions currently renders 72 inches wide |
| `feat/aggregate` | `scripts/08_aggregate.py` — reads all `results/*/`, emits cross-run tables + multi-seed mean±std | nothing currently combines runs |
| **`feat/rsna-loader`** | **DICOM loader (`pydicom`), own train/val/test splits, label mapping** | **On the critical path — external datasets are the priority, and RSNA cannot start without it.** Do this first. |
| `feat/efficiency-metrics` | MACs/FLOPs via `thop`/`fvcore`, peak memory, CPU-latency flag | Phase C; independent, can land during Wave 1 |

Merge all into `main`, **then** cut every `exp/*` branch from that commit so all
runs share identical code.

> **Do this on day 1, it costs nothing but waiting:** submit the CheXpert
> Research Use Agreement. It is pure latency, not work, and CheXpert is one of
> only two genuinely independent second external corpora (see Wave 2).

---

## Wave 1 — external data first, everything else alongside

**Priority: the non-MedMNIST datasets.** The MedMNIST siblings test
*generality* (new task, retrain from scratch); only a real external chest X-ray
corpus tests *transfer*, and transfer is the claim §0 invalidated. So RSNA leads
and the siblings move to Wave 2.

Nothing here depends on anything else — each run trains its own frozen Baseline 1
in its own root, so there is no cross-run ordering. Comparisons happen at
aggregation time.

| Priority | Branch | Runner | `config.py` diff | Wall clock |
|---|---|---|---|---|
| **1** | `exp/rsna-transfer` | **Kaggle interactive (GPU)** | none — trains on PneumoniaMNIST, *evaluates* on RSNA | ~1.5 h |
| **2** | `exp/rsna-full` | **Kaggle commit #1 (GPU)** | `DATA_FLAG="rsna"` — trains *and* evaluates on RSNA | ~5 h |
| 3 | `exp/pneumonia-main` | Colab A (GPU) | full grid: all 13 corruptions, `EVAL_SEVERITIES=[0..4]` | ~4 h |
| 4 | `exp/seed1` | Colab B (GPU) | `SEED = 1` | ~2.5 h |
| 5 | `exp/seed2` | Colab C (GPU) | `SEED = 2` | ~2.5 h |
| 6 | `exp/pneumonia-mse-only` | Kaggle commit #2 | `AE_LAMBDA_MAX = 0.0` | ~1.5 h |

### Why RSNA gets two branches

Two genuinely different experiments, and the cheap one is also the more
important one:

- **`exp/rsna-transfer`** — train on PneumoniaMNIST (pediatric), evaluate on RSNA
  (adult). This is the **real Sub-Q2**, replacing the invalidated Kermany claim.
  Nearly free once the loader exists: no training at all, it reuses the Wave-0
  checkpoints. **Do this one first.**
- **`exp/rsna-full`** — train *and* evaluate on RSNA, full corruption grid. RSNA
  has 30k exams, plenty to train on, so this answers "does the whole story hold
  on a real adult clinical dataset" — a stronger version of what a MedMNIST
  sibling would have shown.

Expect a large absolute accuracy drop on the transfer run (pediatric → adult,
different preprocessing). **That is expected and fine — the quantity of interest
is recovery gain, not absolute accuracy.** Frame it that way in the paper from
the start or the result reads as failure.

### Seeds: what to look for

`exp/pneumonia-main` + `exp/seed1` + `exp/seed2` give three seeds. Aggregate
`mean_gain` per width and compare **between-width spread against between-seed
spread**. If seed noise is larger, "the smallest module wins" is unsupported and
slide 7 needs rewording. Highest-value statistical result in Wave 1.

---

## Wave 2 — second external corpus, then the MedMNIST siblings

### ⚠️ RSNA and NIH ChestX-ray14 are not independent

**RSNA's 30k exams are a subset of NIH ChestX-ray14.** Running both gives you one
dataset twice, not two external validations. For a genuinely independent second
corpus the options are **CheXpert** (Stanford, different institution) or
**PadChest** (Spain, different institution *and* geography).

| Priority | Branch | Dataset | Access | Note |
|---|---|---|---|---|
| 1 | `exp/padchest-transfer` | PadChest | openly accessible from BIMCV | No agreement to wait on — best second external set for a course timeline. Adds a geography shift. |
| 2 | `exp/chexpert-transfer` | CheXpert | Stanford RUA (submit day 1) | Runnable whenever the agreement clears. |
| 3 | `feat/chestmnist-adapter` → `exp/chest-pneumonia` | ChestMNIST, pneumonia label only | via `medmnist` | Adult NIH population with full registry support and no DICOM work — but multi-label, and the pneumonia label is rare and NLP-mined. |
| 4 | `exp/blood-main` | BloodMNIST | via `medmnist` | The balanced-class control: does collapse survive without a majority class? Still worth doing — just not before the external work. |
| 5 | `exp/derma-main` | DermaMNIST | via `medmnist` | Delivers the slide-9 dermatology promise. |

### Slide-9 branches (independent of dataset work — run any time)

Both address "classifier-agnostic recovery," slide 9's most interesting promise.
See [`PRESENTATION_FEEDBACK.md`](PRESENTATION_FEEDBACK.md) §4 for the reasoning.

| Branch | Change | Cost | Delivers |
|---|---|---|---|
| `exp/coupling-matrix` | train an AE guided by **B2** instead of B1 | 1 AE run + 2 evals | Fills the missing half of the 2×2 coupling table. You already have the other half. Turns the coupling limitation into numbers. |
| `feat/feature-loss` → `exp/feature-guided` | `train_recovery(guidance="features")` — MSE on frozen ImageNet backbone features instead of CE | small code change + λ sweep | Classifier-agnostic *by construction* and label-free. If gains hold on both classifiers, slide 9's promise ships as a result rather than future work. |

`exp/coupling-matrix` needs no new code and no new dataset — good filler for a
spare Colab account while the big grids run.
| — | *(verify only)* | COVID-19 Radiography | Kaggle | **Do not use as a dataset.** Hash-check against Kermany first — its 1,345 viral-pneumonia images appear to be Kermany's. See `PRESENTATION_FEEDBACK.md` §1. |

### Do external runs on Kaggle

**The datasets are already there** — no download, no API token, no 3.6 GB
transfer. Just *Add Data* and read from `/kaggle/input/`:

| Dataset | Kaggle source |
|---|---|
| RSNA Pneumonia Detection | `rsna-pneumonia-detection-challenge` (competition data) |
| NIH ChestX-ray14 | `nih-chest-xrays/data` |
| COVID-19 Radiography | `tawsifurrahman/covid19-radiography-database` |

RSNA specifics for `feat/rsna-loader`: DICOM via `pydicom`, single-channel → RGB
→ 64×64 to match `IMAGE_SIZE`, and it borrows PneumoniaMNIST's corruption
registry (the chest-X-ray-appropriate set) — exactly what `data.Kermany` already
does. Label mapping: `Normal` → 0, `Lung Opacity` → 1, **discard** the ambiguous
`No Lung Opacity / Not Normal` group, and record how many images that dropped.

---

## Platform notes

### Kaggle

- **12 h** per session (CPU or GPU); **~30 h/week** GPU quota, floating upward
  with demand.
- **Concurrency: 1 interactive GPU session + 2 commit sessions.** That is the
  parallelism multiplier — *Save Version → Save & Run All (Commit)* runs
  detached, so you can close the tab.
- **32 GB RAM** (vs free Colab's ~13), and more vCPUs than free Colab's ~2 —
  which matters a lot here, since corruption rendering is CPU-bound. **Prefer
  Kaggle for the full 13×5 grid runs.**
- `/kaggle/working` is 20 GB and persists with the notebook version.
- **Turn Internet ON** in notebook settings, or `git clone` and `pip install`
  both fail.
- No Google Drive. Results go back via git (below) or as notebook output.

### Colab

- Set `ROBUSTMED_ROOT` to a Drive path **before** any `robustmed` import —
  `config.ROOT` resolves at import time.
- One Drive folder per run even on separate accounts, so results stay
  distinguishable if you later consolidate.

### Pushing results back from either platform

Use a **fine-grained personal access token scoped to this one repository**,
stored in the platform's secret store — Kaggle: *Add-ons → Secrets*; Colab:
*userdata / Secrets panel*. Never paste a token into a notebook cell, and never
commit one: this repo is public, and a committed token is compromised the moment
it is pushed.

```python
# Kaggle
from kaggle_secrets import UserSecretsClient
TOKEN = UserSecretsClient().get_secret("GH_PAT")

# Colab
from google.colab import userdata
TOKEN = userdata.get("GH_PAT")
```

```python
import subprocess, os
BRANCH = "exp/blood-main"
subprocess.run(f'git config user.email "you@example.com"', shell=True)
subprocess.run(f'git config user.name "runner"', shell=True)
subprocess.run("python scripts/export_results.py", shell=True)   # from feat/results-export
subprocess.run("git add results/", shell=True)
subprocess.run(f'git commit -m "results: {BRANCH}"', shell=True)
subprocess.run(
    f"git push https://{TOKEN}@github.com/FahimSadikRashad/"
    f"medical-lightweight-recovery-pipeline.git HEAD:{BRANCH}",
    shell=True)
```

If you would rather not put a token on Kaggle at all: *Save Version* keeps the
notebook's output files, and you can download `results/` manually. Slower, but
zero credential exposure.

---

## Timeline

| Stage | Who | Wall clock |
|---|---|---|
| Day 1 — submit CheXpert RUA | you, 15 min | pure latency, start it immediately |
| Wave 0 — code changes (`feat/rsna-loader` first) | local, no GPU | a few hours |
| Wave 1 — 6 runs in parallel | 3 Colab + 3 Kaggle sessions | ~5 h |
| Aggregate + read results | local | ~1 h |
| Wave 2 — PadChest, then MedMNIST siblings | Kaggle + Colab | ~1 day |

Wave 1 answers all three remarks: **remark 1** via RSNA (a real external adult
corpus, replacing the invalidated Kermany claim), **remark 3** via the full 13×5
corruption grid, and the multi-seed error bars slide 7 currently lacks.
**Remark 2** (efficiency) lands via `feat/efficiency-metrics` and needs no GPU at
all — it can run while everything else is training.

The critical path is `feat/rsna-loader`. Everything in Wave 1 except the RSNA
branches could start today; the RSNA branches cannot start until it lands, and
they are the priority. So that loader is the first thing to build.

---

## Sources

- [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle concurrent GPU session limits](https://www.kaggle.com/discussions/general/105509)
- [Kaggle floating GPU quota](https://www.kaggle.com/product-feedback/173129)
- [Enabling internet access on Kaggle kernels](https://www.kaggle.com/product-feedback/63544)
- [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview)
- [NIH ChestX-ray14 on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
