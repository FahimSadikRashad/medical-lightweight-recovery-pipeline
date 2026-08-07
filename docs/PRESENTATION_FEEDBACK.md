# Progress-presentation feedback → action plan

Source: `CSE6207_Progress_Presentation .pptx` (10 slides), plus three remarks
raised at the presentation. This file turns both into concrete work items.

> **Before running any of this**, see [`RELATED_WORK.md`](RELATED_WORK.md).
> Classifier-guided restoration has substantial prior art (task-driven /
> recognition-aware restoration, from 2018 onward), and Decorruptor (ECCV 2024)
> has already made the efficiency argument in this space. Neither sinks the
> project, but both change how the contribution must be framed — and it is
> cheaper to know now than after the tables are written.

**Remarks received:**
1. Test on another dataset.
2. For the "lightweight" claim, show efficiency metrics — latency or similar.
3. Compare against the *full* set of corruptions and baselines.

**Slide 9 "Next Direction" (your own list):** position against heavy diffusion
recovery; classifier-agnostic recovery via feature-space loss; other lightweight
recovery modules (DnCNN, NAFNet, HINet); other imaging modalities.

---

## 0. Blocking issue found while researching remark 1

**Sub-Q2 is not a cross-domain transfer experiment.** Slide 8 reports "Real
Kermany X-rays: transfer gain +0.232, identical to in-domain." The reason it is
identical is that it *is* in-domain.

PneumoniaMNIST is **derived from** the Kermany pediatric chest X-ray dataset —
5,856 images, split 4,708 train / 524 val / 624 test. Kermany's Kaggle release
is 5,216 train / 16 val / 624 test. The arithmetic is decisive: PneumoniaMNIST's
624-image test split **is** Kermany's `test/` folder, just at native resolution
instead of 64×64.

Independent confirmation from your own run: Baseline 1's raw accuracy on every
collapsed condition was exactly **0.6250 = 390/624** — precisely the class
composition of Kermany's `test/` directory (234 NORMAL / 390 PNEUMONIA).

`config.KERMANY_TEST_DIR` points at that folder. So `06_generalization.py`
Sub-Q2 evaluates **the same 624 images** the in-domain table already uses.

What it actually measures: robustness to resolution and preprocessing
(full-res → 64×64 resize) on identical images. That is a legitimate ablation,
but it is not domain transfer, and the slide's claim must be withdrawn.

**Actions:**
- [ ] Relabel Sub-Q2 as a *resolution/preprocessing* ablation, not transfer.
- [ ] Verify the overlap directly: hash the 624 Kermany test images, downsample
      to 64×64, and compare against PneumoniaMNIST's test tensors.
- [ ] Get a genuine external dataset (see §1). This is now remark 1's real content.

> **Label mapping verified — no action needed.** `medmnist/info.py` gives
> PneumoniaMNIST `{"0": "normal", "1": "pneumonia"}`, so `data.Kermany`'s
> NORMAL=0 / PNEUMONIA=1 is correct and Sub-Q2's numbers are not inverted.
>
> Watch this per dataset though: BreastMNIST is `{"0": "malignant", "1": "normal,
> benign"}` — the pathology is class 0 there, the opposite convention.

---

## 1. Remark 1 — another dataset

MedMNIST-C covers **12 datasets across 9 modalities**, all with the same
corruption API you already use: `pathmnist`, `bloodmnist`, `dermamnist`,
`retinamnist`, `tissuemnist`, `octmnist`, `breastmnist`, `chestmnist`,
`pneumoniamnist`, `organamnist`, `organcmnist`, `organsmnist`.

Because `config.DATA_FLAG` already drives everything, adding a dataset is a
config change plus a rerun — no new code.

### Which to pick, and why

Do not pick on modality diversity alone. Pick datasets that **test whether your
central mechanism is real**. Your collapse story depends on class imbalance —
"collapse to majority" is only meaningful when there *is* a majority.

Because the **full** suite (train both baselines, train the recovery sweep, run
the whole corruption grid, run every ablation) will be repeated on the new
dataset, test-split size matters: it sets the noise floor on every number in the
paper. Exact counts, from `medmnist/info.py`:

| Dataset | Modality | Task | Ch | train / val / **test** | Verdict for a full run |
|---|---|---|---|---|---|
| **bloodmnist** | microscopy | 8-class, ~balanced | 3 | 11,959 / 1,712 / **3,421** | **Primary pick.** The critical control: if collapse still happens with balanced classes it is a genuine representation failure; if not, your finding is partly an imbalance artifact. Largest usable test set, natively RGB. |
| **dermamnist** | dermatoscopy | 7-class, heavily imbalanced | 3 | 7,007 / 1,003 / **2,005** | **Second pick.** Slide 9 promised dermatology. Imbalanced like pneumonia, so the collapse-to-majority story should transfer. Natively RGB, so it also tests whether recovery depends on your grayscale→RGB duplication. |
| **chestmnist** (pneumonia label only) | chest X-ray, **adult** | multi-label ×14 | 1 | 78,468 / 11,219 / **22,433** | **Most interesting.** Same task, same modality, *different population and source* (NIH, adult) — and it has full MedMNIST-C registry support, so no DICOM work. Caveats below. |
| ~~breastmnist~~ | ultrasound | binary, imbalanced | 1 | 546 / 78 / **156** | **Downgraded — do not use for paper claims.** A 156-image test split gives ~50–100 images per class; every balanced-accuracy point would carry enormous variance, and you would be reading noise across 65 conditions. Fine as a fast smoke test of the code path, nothing more. |

**Recommended: `bloodmnist` as the primary full run**, `dermamnist` second.

### The chestmnist option

Tempting because it is the only same-task, same-modality, different-population
dataset with native MedMNIST-C support. Two real obstacles:

- **It is multi-label** (`task = "multi-label, binary-class"`, 14 labels,
  pneumonia at index 6). Your pipeline is `CrossEntropyLoss` + `argmax`, so this
  needs an adapter that slices label 6 into a single binary target. Contained
  change, but it is a change.
- **The pneumonia positive rate is very low** (a few percent in NIH
  ChestX-ray14) and the labels are NLP-mined from reports, so they are noisy.
  Extreme imbalance plus label noise may make the binary task too hard to show
  anything clean.

Worth attempting *after* bloodmnist, not instead of it. Also subsample the
22,433-image test split — 65 conditions × 22k is ~1.5M forward passes per model.

**Actions:**
- [ ] Make result paths dataset-scoped so runs don't overwrite each other:
      `RESULT_DIR = ROOT / "results" / DATA_FLAG`. **Do this before the first
      extra run**, or it silently clobbers the PneumoniaMNIST results you already
      have.
- [ ] Full suite (stages 01–07) with `DATA_FLAG = "bloodmnist"`.
- [ ] Full suite with `DATA_FLAG = "dermamnist"`.
- [ ] Optional: chestmnist with a pneumonia-label adapter + test subsampling.
- [ ] Report per-dataset, and state plainly whether collapse reproduces.
- [ ] Note in the paper that `AE_WIDTHS` may need rescaling for 3-channel input —
      the w=4 parameter count changes slightly, so re-quote it per dataset rather
      than reusing "1,119".

> Multi-class changes the metric story: with 8 classes, chance balanced accuracy
> is 0.125, not 0.50, and "collapse to one class" is a much larger drop. Check
> `engine.collapsed()` still reads correctly — it counts distinct predictions, so
> it generalizes, but the *interpretation* of a 0.125 floor needs a sentence.

### Non-MedMNIST options — for genuine cross-domain transfer

> **These take priority over the MedMNIST siblings above.** The siblings test
> generality; only a real external chest X-ray corpus tests transfer, which is
> the claim §0 invalidated. Execution order is in
> [`PARALLEL_PLAN.md`](PARALLEL_PLAN.md) — RSNA leads Wave 1, siblings move to
> Wave 2.
>
> **RSNA ⊂ NIH ChestX-ray14.** Running both is one dataset twice, not two
> external validations. An independent *second* corpus means CheXpert (different
> institution) or PadChest (different institution and country).

The MedMNIST siblings above test **generality** (new task, retrain everything).
They do not test **transfer**, which needs the *same* binary pneumonia task on a
different acquisition and population. Since §0 removed Kermany as an external
set, this is the only way to recover the Sub-Q2 claim.

A useful property of the pneumonia task: PneumoniaMNIST is **pediatric**
(Kermany, ages 1–5), while every large public chest X-ray corpus is **adult**.
Pediatric → adult is a substantial, honest domain shift, so a transfer result
here is worth considerably more than the current one.

| Dataset | Size / labels | Access | Verdict |
|---|---|---|---|
| **RSNA Pneumonia Detection Challenge** | 30k exams (15k pneumonia-positive, 15k negative), DICOM, adult | Kaggle competition data; academic use permitted with attribution | **Best fit.** Real pneumonia labels, binary-izable, independent of Kermany, and small enough to handle. Start here. |
| NIH ChestX-ray14 | 112,120 images, 14 labels incl. Pneumonia, adult | Open on Kaggle | Superset that RSNA is drawn from. Pneumonia labels are NLP-mined from reports and noisy; ~45 GB. Use only if RSNA is unavailable. |
| CheXpert | 224,316 images, 14 observations incl. Pneumonia, adult | Requires signing a Stanford Research Use Agreement | Good data, but the agreement adds lead time. Research-use only, no commercial. |
| PadChest | ~160k images, multi-label, Spanish reports, adult | Openly accessible from BIMCV | Viable; a second geography is a nice extra shift. Large. |
| ⚠️ **COVID-19 Radiography Database** | 10,192 normal / 1,345 viral pneumonia / 3,616 COVID | Open on Kaggle | **Avoid — contamination risk.** Its viral-pneumonia count is exactly **1,345**, identical to Kermany's viral-pneumonia count. That subset is sourced from Kermany, so this would reproduce the §0 problem. Verify by hashing before using any part of it. |

**Recommended: RSNA.** Practical notes for wiring it up:

- Images are **DICOM** — needs `pydicom`; `data.Kermany` assumes PIL-readable
  files, so this needs a sibling loader (single-channel → RGB → 64×64, matching
  `IMAGE_SIZE`).
- ~3.6 GB for the training image set. Fine on Colab; you only need a *test* set,
  so subsampling ~1,000 per class is plenty.
- Label mapping for a clean binary comparable to yours: the detailed class file
  has `Normal`, `Lung Opacity`, and `No Lung Opacity / Not Normal`. Use
  **`Normal` → 0** and **`Lung Opacity` → 1**, and **discard the ambiguous third
  group**. State this in the paper — it is a defensible choice, but it must be
  explicit, since including the middle group would change the numbers.
- **Expect a large absolute accuracy drop** (pediatric → adult, plus different
  preprocessing). That is fine and expected. The quantity of interest is the
  **recovery gain**, not absolute accuracy — frame it that way from the start, or
  the result reads as failure.

**Actions:**
- [ ] Download RSNA via Kaggle; add a DICOM loader alongside `data.Kermany`.
- [ ] Map labels as above; record how many images the middle group discarded.
- [ ] Report no-recovery vs. with-recovery gain, clean and corrupted.
- [ ] Hash-check any COVID-19 Radiography subset against Kermany before use.

---

## 2. Remark 2 — efficiency metrics for the "lightweight" claim

Two gaps here, and the second is worse than the first.

### Gap A: the metrics themselves are thin

You currently report parameter count and (after the restructure) GPU latency.
Standard practice for an efficiency claim is params + FLOPs/MACs + measured
latency + peak memory, with the explicit caveat that **FLOPs are not predictive
of latency** — a low-FLOP model can be memory-bound and slower in practice, so
measured wall-clock on target hardware is required, not optional.

| Metric | How | Note |
|---|---|---|
| Parameters | already have (`models.count_params`) | w=4 → **1,119** |
| MACs / FLOPs | `thop`, `ptflops`, or `fvcore` | report MACs; say which |
| Latency | already have (`engine.latency_ms`, CUDA events) | needs CPU too — see Gap B |
| Throughput | images/sec at batch 1 and 128 | batch-1 is the deployment case |
| Peak memory | `torch.cuda.max_memory_allocated` / `torch.profiler` | |
| Energy | skip unless you have a power meter | honest omission > fabricated estimate |

### Gap B: "lightweight" is currently an unanchored claim

You measure your module against **other sizes of itself** (w=4/8/16/32). Nothing
in the current experiments says 1,119 params is *light* — light compared to
what? Two anchors, cheapest first:

**Anchor 1 (free, do this regardless): express cost as a fraction of the
classifier you are protecting.** MobileNetV2 is 2.23M params, so the recovery
module is **0.05% of the classifier's parameters**. Report the same ratio for
MACs and for latency — "recovery adds X% to end-to-end inference." This is
compelling, requires no new baseline, and directly supports the slide-2 framing.

**Anchor 2 (does the real work): a heavy recovery baseline.** Slide 9 already
says "position against heavy diffusion recovery." Until a diffusion-based
recovery method is in the table at its real parameter count and latency, "how
light can recovery be?" has no upper reference point. You do not need to match
its accuracy — you need its *cost*, to show the ratio.

### Gap C: your title says "Constrained Hardware" — you measured on a T4

The presentation title is "…under Constrained Hardware." Every timing you have
is from a Colab T4 GPU. A T4 is not constrained hardware, and a reviewer will
say so.

Cheapest credible fix: **measure CPU-only, single-thread** as the constrained
proxy (`torch.set_num_threads(1)`, `device="cpu"`). `engine.latency_ms` already
falls back to `perf_counter` on CPU, so this is a flag away. If you can borrow a
Raspberry Pi or Jetson Nano, one real measurement there is worth more than any
amount of GPU timing.

**Actions:**
- [ ] Add MACs/FLOPs to `engine`, extend `RECOVERY_COST` payload.
- [ ] Report recovery cost as a **fraction of classifier cost** (params, MACs, latency).
- [ ] Add CPU single-thread latency alongside GPU. Label which is which.
- [ ] Add at least one heavy recovery baseline for cost anchoring.
- [ ] Batch-1 throughput, since that is the deployment condition.

---

## 3. Remark 3 — all corruptions and all baselines

### Corruptions: you are testing 3 of 13

`CORRUPTIONS_DS["pneumoniamnist"]` contains **13** corruptions:

```
pixelate            jpeg_compression                       <- digital
gaussian_noise      speckle_noise    impulse_noise   shot_noise   <- noise
gaussian_blur                                          <- blur
brightness_up       brightness_down                    <- photometric
contrast_up         contrast_down
gamma_corr_up       gamma_corr_down
```

`config.EVAL_CORRUPTIONS` covers `gaussian_noise`, `gaussian_blur`,
`jpeg_compression` — **3 of 13**, at 3 of 5 severities. Full grid is 65
conditions; you report 9. A reviewer will ask whether the three were chosen
after seeing results, and you currently have no answer.

**The specific risk, not just a coverage complaint.** All three tested
corruptions are *degradations* — they destroy information. The six untested
photometric corruptions (brightness/contrast/gamma, up and down) are
**invertible intensity transforms** that destroy almost nothing. Your recovery
AE was trained only on degradations, and its architecture ends in a
`clamp(0,1)` with no global intensity pathway. There is a live possibility that
it does nothing useful — or actively harms — on photometric shifts.

Better to find that yourself than in review. It is also a *good* result if
handled honestly: "recovery addresses information-destroying corruptions;
photometric shifts need normalization instead" is a clean, defensible scope
statement, and it's a stronger paper than an unqualified claim.

Evaluation is cheap — 65 conditions × 624 images, no training. Run it.

### Baselines: two obvious gaps

Current: Baseline 1 (frozen), Baseline 2 (augmentation), box denoiser.

| Missing baseline | Why it matters |
|---|---|
| **Heavy/diffusion recovery** | Upper reference for both accuracy and cost. Slide 9 already commits to this. Without it "how light can recovery be" is unanchored. |
| **Other lightweight modules** — but **at matched parameter budget** | Right now the comparison is your AE vs. a 0-param box filter. That gap is easy to win and proves little. Slide 9's list (DnCNN/NAFNet/HINet) is also dated and, more importantly, *scale-mismatched* — those are millions of params against your 1,119. The experiment that actually answers "how light can recovery be?" is **architecture comparison at a fixed ~1k budget**: scale DnCNN, a NAFNet block, and a Mamba block down and see which uses the budget best. Nobody has run that. See [`RELATED_WORK.md`](RELATED_WORK.md) Tier 3. |
| **Test-time adaptation** (TENT/MEMO) | Optional, but these also improve corrupted accuracy without retraining on clean data, so a reviewer may raise them as the natural competitor to "no retraining." |
| **MSE-only AE** | You already know it fails. It is your cleanest evidence that the perceptual loss is the mechanism — but it is not in any table. Nearly free: `AE_LAMBDA_MAX = 0`. |

**Actions:**
- [ ] `EVAL_CORRUPTIONS = corruptions.names()`; `EVAL_SEVERITIES = [0,1,2,3,4]`.
- [ ] Report grouped by category (digital / noise / blur / photometric), not as 65 flat rows.
- [ ] Retrain recovery on a corruption subset spanning categories, so photometric
      shifts are represented in training — then compare against the
      degradation-only module. This turns the risk above into a contribution.
- [ ] Add the MSE-only row.
- [ ] Add one real lightweight denoiser baseline.
- [ ] Add one heavy recovery baseline for cost anchoring.

---

## 4. Slide 9 coverage map

Every "Next Direction" you promised, and where it lands:

| Slide 9 promise | Status | Covered by |
|---|---|---|
| Other imaging modalities (dermatology, retinal, histology) | **Scheduled** | §1 — `dermamnist` / `retinamnist` / `pathmnist`, Wave 2 |
| Other lightweight recovery modules (DnCNN, NAFNet, HINet) | **Scheduled** | §3 baselines table |
| Position against heavy diffusion recovery | **Scheduled** | §2 Anchor 2 — needed for the cost claim regardless |
| Classifier-agnostic recovery (feature-space loss) | **Partly deliverable now** | below |

### Classifier-agnostic recovery — three tiers, two of them fit the timeline

An earlier draft of this file called this "future work unless the timeline is
generous." That was wrong: you already have the pieces for the first two tiers.

#### Tier 1 — measure the coupling properly (nearly free)

Right now "recovery is classifier-coupled" is an ad-hoc observation from a
throwaway cell. It should be a 2×2 table, and **you already have three of the
four cells** — because Baseline 1 and Baseline 2 are two independently trained
classifiers:

| | eval on B1 (clean-trained) | eval on B2 (aug-trained) |
|---|---|---|
| AE trained against **B1** | ✅ stage 04 | ✅ stage 05 stacked row |
| AE trained against **B2** | ❌ missing | ❌ missing |

Cost: **one extra AE training run** (guide `train_recovery` with B2 instead of
B1), then two evaluations. That converts a hand-wave into a documented
limitation with numbers — which is what a reviewer will want whether or not you
fix it.

#### Tier 2 — classifier-agnostic guidance (a small, real contribution)

The coupling comes from the CE term binding the AE to one classifier's decision
boundary *and* to labels. Replace it with a **feature-space loss against a frozen,
task-agnostic encoder**:

```
loss = MSE(recon, clean) + λ · MSE( φ(recon), φ(clean) )
```

where `φ` is the **ImageNet-pretrained backbone features** — not the fine-tuned
classifier. Two properties that matter:

- **Classifier-agnostic by construction.** `φ` is shared and never task-tuned, so
  the module is not bound to any downstream head. No new trained model needed —
  `models.backbone(pretrained=True)` already gives you `φ`.
- **Label-free.** No `y` in the loss, so recovery could be trained on unlabelled
  images — a genuinely useful property worth stating.

Implementation is contained: hook MobileNetV2's `features` module output, global
average pool, MSE. Then rerun the **same Tier-1 2×2**. If gains hold on *both*
classifiers, slide 9's promise is delivered with evidence, not deferred.

> Retune λ. Feature MSE has a completely different magnitude from CE, so
> `AE_LAMBDA_MAX = 1.5` will not transfer — sweep it, or the method will look
> like it failed when it was only mis-scaled.

#### Tier 3 — genuine future work

Multi-classifier ensemble guidance, learned adapters between recovery and
arbitrary heads, theory for when a projection-to-clean-manifold is
classifier-independent. This is a paper of its own. Say so on the slide.

### What to put on the slide

Tiers 1 and 2 move "classifier-agnostic recovery" from *promised future work* to
*a result with a table*. That is a materially stronger talk, and Tier 1 alone
costs one training run. Do Tier 1 regardless; do Tier 2 if you want the
contribution rather than just the caveat.

**Actions:**
- [ ] Tier 1: train an AE guided by B2, fill in the 2×2 coupling table.
- [ ] Tier 2: add a feature-space loss option to `engine.train_recovery`
      (`guidance="ce" | "features"`), sweep λ, rerun the 2×2.
- [ ] Report the 2×2 in the paper whichever way it comes out — a documented
      limitation beats an undocumented one.

---

## 5. Suggested order

Cheapest-and-most-load-bearing first. Items 1–3 are hours, not days, and 1 and 2
both protect you from claims that are currently wrong.

1. **Relabel Sub-Q2** (§0). A stated claim is currently incorrect. Free.
2. **Dataset-scoped result paths** (§1). Do before any extra run or you lose data.
3. **Cost-as-fraction-of-classifier + CPU latency** (§2). Hours, and directly
   answers remark 2 with what you already have.
4. **Full 13×5 corruption evaluation** (§3). Eval only, no training. Answers
   remark 3's first half, and surfaces the photometric risk.
5. **MSE-only row** (§3). One config flag, converts a known result into evidence.
6. **breastmnist + bloodmnist** (§1). Answers remark 1 properly. bloodmnist is
   the scientific control — do not skip it for the easier one.
7. **Multi-seed reruns** (from `EXPERIMENTS.md`) — still the top statistical gap;
   "smallest module wins" has no error bars, and it is slide 7's whole claim.
8. **Lightweight + heavy recovery baselines** (§2, §3). Most work; do last.

---

## Sources

- [MedMNIST-C: Comprehensive benchmark and improved classifier robustness (arXiv:2406.17536)](https://arxiv.org/abs/2406.17536)
- [medmnistc-api — corruption registry](https://github.com/francescodisalvo05/medmnistc-api)
- [MedMNIST-C dataset page, Uni Bamberg](https://www.uni-bamberg.de/en/ai/chair-of-explainable-machine-learning/software-datasets/dataset-medmnist-c-12-corrupted-benchmark-datasets-and-augmentation-apis-for-robust-medical-image-classification/)
- [MedMNIST v2 (Scientific Data) — PneumoniaMNIST provenance and splits](https://www.nature.com/articles/s41597-022-01721-8)
- [Efficient Processing of Deep Neural Networks (Sze et al.) — why FLOPs ≠ latency](https://eyeriss.mit.edu/2020_efficient_dnn_excerpt.pdf)
- [Lightweight Deep Learning for Resource-Constrained Environments: A Survey (ACM CSUR)](https://dl.acm.org/doi/10.1145/3657282)
- [The Framework Tax: Disparities Between Inference Efficiency in Research and Deployment (arXiv:2302.06117)](https://arxiv.org/pdf/2302.06117)
