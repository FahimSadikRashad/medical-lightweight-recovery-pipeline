# Work plan — efficient recovery modules across medical modalities

Supersedes the scope sections of `PARALLEL_PLAN.md`. Reads on top of
`RELATED_WORK.md` (novelty risk) and `PRESENTATION_FEEDBACK.md` (remarks 1–3).

**Revision note.** An earlier draft of this file centred the ConvAE and proposed
MedMNIST siblings as the second dataset. Both were wrong. The ConvAE is a naive
baseline, not the subject; and staying inside the MedMNIST family tests almost
nothing about generality — same preprocessing, same 64×64 resolution, same
release pipeline, and in PneumoniaMNIST's case a known provenance problem
(`PRESENTATION_FEEDBACK.md` §0).

---

## 0. Framing

**Research question.** Among tiny, cheap input-space modules placed in front of a
**frozen** medical classifier, which design delivers corruption robustness that
holds up across every condition and every modality — and why?

Parameter count is a reported cost, not the claim.

### Contributions

**K1 — Benchmark.** First systematic comparison of efficient restoration
architectures used as front-ends to frozen medical classifiers, evaluated on
worst-case rather than mean performance, across modalities. This is the spine of
the paper. The ConvAE is one arm in it, labelled as the naive baseline.

**K2 — Finding: restoration objectives conflict in the compute-bounded regime.**
Noise and compression want low-pass behaviour; blur wants high-pass. A full-size
restoration network has capacity to learn both conditionally. A tiny module does
not, so it compromises — and the compromise is measurable in the data we already
have.

> ⚠️ **Provisional until Stage 1, and B1 says it is probably too narrow.**
> Everything below is derived from **3 of the pneumoniamnist registry's 13
> families**. Of the 10 we never train on, **6 are photometric** — brightness ±,
> contrast ±, gamma ±. Photometric corruption is a global intensity remap, and a
> module built from spatial convolutions has no mechanism for it: neither
> low-pass nor high-pass filtering corrects a gamma shift.
>
> **Revised hypothesis to test in Stage 1:** the conflict is *three*-way, not
> two-way — denoise (low-pass), deblur (high-pass), photometric (intensity
> remap) — and the worst conditions will be photometric rather than blur.
> If that holds, K2 gets stronger and K3 gains a third branch (a per-channel
> affine/gamma correction, ~6 parameters). Stage 1 prints the category table
> that settles it.

| condition | severity slope (sev4 − sev0) | status |
|---|---|---|
| gaussian_noise | −0.004 | saturated — recovery reaches clean-level |
| jpeg_compression | +0.003 | saturated |
| **gaussian_blur** | **−0.047, worse in 11/11 runs** | **binding constraint** |

`gaussian_blur_sev4` is the worst condition in 10 of 11 trained runs. Remaining
headroom is ~0.035 on noise/JPEG versus ~0.083 on strong blur — 2.4× larger.
Capacity does not buy the difference: conditional on training, w=4 gives +0.221
and w=32 gives +0.175 across a 48× parameter range.

**K3 — Method.** A gated module that resolves the conflict by routing rather than
by adding capacity (§3).

**Appendix finding — output parameterization governs trainability at small
scale.** 9 of 20 runs in the stability sweep produced no usable model, because
`torch.clamp(out, 0, 1)` has zero gradient outside `[0,1]` and a non-residual
decoder can initialise negative everywhere. Reproduced: seeds 1/w=4 and 2/w=8
show 100% negative pre-clamp output and gradient exactly `0.00e+00` — precisely
the two runs whose loss froze at `0.3572 + λ·4.2037`. This was co-headline in the
previous draft; it is demoted. It is a bug in a baseline, and its real function
here is **hygiene**: every arm must use a gradient-safe bounded output, or the
module comparison is confounded from the start (most modern restoration
architectures carry a global residual and would be immune, making them look
spuriously more stable than the ConvAE).

### Honest novelty risk

Degradation-aware and blind restoration (DASR lineage) already condition
restoration on estimated degradation. The differentiator is **not** "conditioning
is new." It is (a) the measured demonstration that the conflict binds
specifically in the compute-bounded regime, and (b) that a ~100-parameter gate
resolves it. Say so in related work rather than letting a reviewer say it.

---

## 1. Two blocking checks

Neither costs more than an hour, and both can invalidate large parts of §4.
Nothing else starts until they are answered.

- [x] **B1 — ANSWERED. Passes, with a condition.** Every one of the 12 registries
      contains at least one blur family, so the cross-modality blur story is
      tellable. But almost nothing transfers *by name*:

      | | |
      |---|---|
      | universal families | only `contrast_down`, `jpeg_compression`, `pixelate` |
      | `gaussian_blur` | **missing from 6/10** registries |
      | `gaussian_noise` | **missing from 5/10** registries |
      | no noise family at all | `pathmnist`, `bloodmnist` |

      **Consequence 1:** every cross-dataset statement must be made at the level
      of *category* (blur / noise / photometric / codec), never corruption name.
      "Blur transfers" is sayable; "gaussian_blur transfers" is not. Implemented
      as `config.CORRUPTION_CATEGORIES` + `config.train_corruptions_for()`, which
      picks one family per category from whatever a registry actually has.

      **Consequence 2:** histology (PCam → `pathmnist`) has **no noise family**,
      so the noise-vs-blur conflict cannot be tested there at all. PCam is
      demoted; ultrasound (`breastmnist`: speckle_noise + motion_blur) and chest
      X-ray both work.
- [ ] **B2 — Resolution vs severity calibration.** MedMNIST-C severities are
      calibrated at MedMNIST resolution. Blur severity is *strongly*
      resolution-dependent — a σ that destroys a 64×64 image barely touches
      1024×1024. Moving to native-resolution clinical corpora therefore breaks the
      calibration unless resolution is held fixed. **Decision needed:** resize every
      corpus to one common resolution (recommended — preserves calibration and
      comparability with existing tables) or re-calibrate severities per corpus
      (defensible, expensive, and a paper subsection of its own).

---

## 2. Metric changes (do first)

The current headline `mean_gain` (`11_stability.py:100`) averages 9 conditions
and answers the wrong question. "Works under any condition" is a floor. Two runs,
near-identical means, 6 points apart on the floor:

| run | mean over corruptions | worst condition | gap |
|---|---|---|---|
| s2_w16 | 0.7921 | 0.7158 | +0.076 |
| s2_w4 | 0.7851 | **0.7765** | +0.009 |

- [ ] Add `worst_gain`, `worst_condition`, `worst_abs`. Promote worst-case to
      headline, keep mean as secondary.
- [ ] Add mCE / relative mCE (Hendrycks & Dietterich). `RELATED_WORK.md` Tier 1
      flags that a robustness reviewer expects it, and it changes what we log.
- [ ] Report `P(train succeeds)` over seeds as a first-class column.
- [ ] Fix `collapsed()` (`engine.py:81`) — `len(pred_counts) == 1` misses `s0_w4`
      at bal=0.5043. Add `balanced < 0.52`. Real failure count is 9, not 8.

---

## 3. Stage plan

### Stage 0 — harness (~1 day, CPU-testable)

Not a research stage. Shared plumbing so every arm and every corpus is treated
identically.

- [ ] **`RecoveryModule` interface**: `[0,1] → [0,1]`, with a single shared
      gradient-safe bounded output (sigmoid, or residual + clamp). Every arm
      inherits it. This is the hygiene fix, applied once, to all modules — not a
      ConvAE repair.
- [ ] Instrument the trainer: log output saturation fraction and gradient norm
      per epoch. Cheap, and it turns the appendix finding into a figure.
- [ ] Dataset abstraction: `data.load_dataset()` returning the same
      `(train, val, test, info)` contract, so non-MedMNIST corpora drop in behind
      one interface. Split `CORRUPTION_REGISTRY_FLAG` from `DATA_FLAG`.
- [ ] Dataset-scope `results/`, `checkpoints/`, `figures/` + a `--dataset` flag,
      so runs cannot clobber each other.
- [ ] Fix `pair_loader` reuse (`11_stability.py:70`) — its generator advances
      across the width loop, so w=4 and w=32 see different data orders within one
      "seed."
- [ ] Re-run the existing 20-run grid as a regression check. Expect failures
      9/20 → ~0/20. If not, the appendix finding is wrong; cheap to know.

### Stage 1 — full corruption characterization (~1h GPU, **evaluation only**)

Presentation remark 3, promoted from the write-up stage to a gate. We currently
train on 3 corruption families at severities 0–2 and evaluate the *same* 3 at
severities 0/2/4. That is far too narrow to support K2 or to choose architectures.

No retraining — this reuses the existing Baseline 1 and recovery checkpoints.

- [ ] Enumerate the full registry (`corruptions.names()`) and widen
      `EVAL_CORRUPTIONS` to all of it, all 5 severities.
- [ ] Report worst-case in **three groups**, because they answer different
      questions and pooling them hides the interesting one:
      | group | meaning |
      |---|---|
      | seen family, seen severity | in-distribution |
      | seen family, unseen severity (4) | current holdout |
      | **unseen family entirely** | the real robustness test — new, and the strongest answer to remark 3 |
- [ ] Re-derive which condition actually binds, and the severity slope per family.
- [ ] Flag degenerate conditions where even the clean-trained baseline sits at
      chance — worst-case over a condition nobody can fix is not informative, and
      it must be reported separately rather than silently dominating the metric.

**Gate:** K2 either survives the full corruption set or is rewritten. If a
non-blur family binds hardest, the Stage 2 arm list changes accordingly — the
whole point of running this before spending GPU on training.

**Separate decision this raises:** do we keep *training* recovery on a narrow set
and evaluate broadly (realistic — you cannot anticipate every fault in
deployment, and it makes unseen-family generalization the headline), or train on
the full set (in-distribution, easier, weaker claim)? Recommend narrow-train /
broad-eval as primary, with one broad-train arm to bound the gap.

### Stage 2 — module comparison, primary corpus (~8h GPU)

**The main experiment.** Sweep each arm across 3 widths — Pareto frontier over
**cost as a fraction of end-to-end inference**, not matched parameters.

| arm | year | design motif being ported | role |
|---|---|---|---|
| Identity | — | — | floor |
| Box filter (exists) | — | fixed low-pass | non-learned low-pass control |
| Fixed unsharp / FFT high-pass | — | fixed high-pass | non-learned high-pass control — the honest control for K2 |
| ConvAE (exists) | — | plain conv encoder-decoder | naive learned baseline |
| DnCNN | 2017 | residual noise prediction | legacy control. Noise is already saturated, so this is deliberately not a contender. |
| NAFNet | 2022 | gated conv, no activation | recognized restoration reference point |
| **SPAN** | 2024 | parameter-free attention | **the field's efficiency reference** — NTIRE 2025 *and* 2026 both require entrants to beat SPAN on runtime, params and FLOPs. Including it makes our efficiency claim commensurable with that literature. |
| **SPANV2** | 2026 | near-pixel branch + depthwise-separable fusion | current NTIRE ESR winner (XiaomiMM). The strongest available "recent" arm. |
| SAFMN *(optional)* | 2023 | spatially-adaptive feature modulation | lightweight-by-design; closest published motif to our gated module |

Selection is principled rather than leaderboard-driven: blur binds, so the
contenders are high-frequency reconstructors, and DnCNN is a control on purpose.

**Where the tiny models actually live.** The NTIRE 2026 *denoising* challenge
reports MambaIR and Restormer as the backbones of choice — both far outside our
regime. The efficient-SR track is where sub-100k architectures are designed on
purpose, which is why the contender list is drawn from there rather than from the
denoising literature. Its 2026 trends — frequency-domain processing, distillation,
custom CUDA kernels — are also directly relevant, and the frequency-domain trend
in particular raises the bar for K3 (§Stage 3): our dual-branch design must be
differentiated from it, not presented as unrelated.

**Scale-gap caveat — state this in the paper.** Every learned arm above was
published at ≥100k parameters. Shrunk to our regime we are not evaluating SPAN or
NAFNet; we are evaluating their *design motifs* at a scale nobody published at.
Claim it as "we port efficient-restoration design motifs to the compute-bounded
regime," never as "we compare against NAFNet." Report both the published
parameter count and ours for every arm.

Two arms deliberately excluded: MambaIR/SSM (needs fragile CUDA kernels; the
latency comparison would be unfair and `PARALLEL_PLAN.md` already flags this) and
Restormer (2022, and used at full scale in NTIRE 2026 — out of regime).

8 arms × 3 widths × 6 seeds ≈ 144 runs.

**Gate:** at least one learned module beats both non-learned controls on
worst-case. If a fixed unsharp filter matches them, that is the finding and the
paper changes shape — far better to learn it here than at Stage 4.

### Stage 3 — the novel module (~4h GPU)

**Multi-branch gated recovery.** A low-pass branch, a high-pass branch, and — if
Stage 1's category table confirms the three-way hypothesis — a photometric branch
(per-channel affine/gamma, ~6 params), combined by a ~100-parameter per-image
gate. Motivated directly by K2: one static filter cannot serve objectives that
call for opposite operations, so predict which one this image needs.

Branch count is decided by Stage 1, not assumed here.

Ablations that make it a contribution rather than a trick:

- [ ] gate vs fixed 50/50 blend — isolates *conditioning* from the two branches
- [ ] learned gate vs oracle gate (true corruption given) — headroom left on the table
- [ ] single-branch controls at matched cost — rules out "it is just bigger"
- [ ] gate output vs true corruption type — does it learn to *identify* the fault?

**Gate:** beats the Stage 2 winner on worst-case, with the ablation showing the
gain comes from conditioning.

### Stage 4 — cross-modality (~8h GPU)

Non-MedMNIST, native clinical corpora, all openly downloadable without Kaggle or
a data-use agreement. Resolution handling per B2.

| corpus | modality | access | n | role |
|---|---|---|---|---|
| **Montgomery + Shenzhen TB** | chest X-ray | direct NLM zips, no auth | 800 | primary non-MNIST. Same modality as our registry, so B1 is satisfied by construction. Real adult clinical data. |
| **BUSI** | breast ultrasound | open (Al-Dhabyani) | 780 | second modality, cheap |
| ~~PCam~~ | histology | HuggingFace | 327k | **dropped by B1** — `pathmnist` has no noise family, so the noise-vs-blur conflict cannot be tested there |
| **NIH ChestX-ray14** | chest X-ray | official NIH Box | 112k | scale, if PCam is blocked by B1. Pull ~3 of 12 tars (~6 GB) |

Recommended: **Montgomery+Shenzhen first**. Two downloads, no auth, full pipeline
feasible with ImageNet-pretrained MobileNetV2 on 800 images, uses the chest X-ray
registry natively, and it independently recovers the transfer claim that
`PRESENTATION_FEEDBACK.md` §0 forced us to withdraw. Highest value per hour in
the plan. Then BUSI for a genuinely different modality.

PneumoniaMNIST is retained only as a **legacy comparability arm** — it is where
all existing numbers live — not as the primary.

Carry the top 3 modules from Stage 3 × 6 seeds: ~18 runs per corpus, not a
full re-grid.

**Gate:** does the module *ranking* transfer, and does blur stay the binding
condition? Ranking stability is the generality claim; matching absolute numbers
across modalities is neither expected nor needed.

### Stage 5 — efficiency and write-up (~1 day)

- [ ] `robustmed/efficiency.py`: MACs via `FlopCounterMode`, CPU single-thread
      latency, GPU latency, throughput at bs=1 and bs=128, peak memory, on-disk
      size. Headline axis: **cost as % of end-to-end classifier inference** —
      operating-point-independent, and it replaces the parameter count.
- [ ] Figures, `FINDINGS.md` refresh, `EXPERIMENTS.md` ledger update.

---

## 4. Cost

≈21h GPU, ≈3 days engineering. Colab-feasible; the existing skip-on-resume logic
already survives disconnects.

Note how cheap the ordering is: Stage 1 costs ~1h of *evaluation* and can
invalidate the premise of the ~20h of training that follows it. Stage 0's
regression re-run is similarly cheap and tests the appendix finding. Both
before any new training starts.

## 5. Risks

| risk | mitigation |
|---|---|
| ~~B1 fails~~ — **resolved**: all registries have blur | Residual risk is naming, not absence: compare by category, never by family name. `config.train_corruptions_for()` enforces this. |
| **B2 unresolved** — severity not comparable across resolution | Fix one common resolution for all corpora. Decide before Stage 4, not after. |
| Module gaps sit inside seed noise | 6 seeds from the start, CIs on worst-case. Already the top statistical problem in `EXPERIMENTS.md`. |
| Gated module reads as DASR-derivative | Position on the finding (K2) and the regime, not the mechanism. |
| **Full corruption set overturns K2** — a non-blur family binds hardest | Exactly why Stage 1 runs before Stage 2. Cost of finding out is ~1h eval; cost of not finding out is the wrong arm list for 144 training runs. |
| Worst-case dominated by an unfixable condition | Report degenerate conditions (baseline already at chance) as a separate group rather than letting them define the floor. |
| Montgomery+Shenzhen too small to train on | 800 images with an ImageNet-pretrained backbone is a standard TB benchmark task; if the classifier is weak, use it test-only for transfer and promote BUSI or NIH to the full-pipeline slot. |

## 6. Decisions needed

1. **B2 — one common resolution, or per-corpus recalibration?** I recommend
   common resolution. (B1 is now answered — see §1.)
2. **Narrow-train / broad-eval, or train on the full corruption set?** I
   recommend narrow-train + broad-eval as primary (it makes unseen-family
   generalization the headline and matches deployment), with one broad-train arm
   to bound the gap.
3. **Corpus order** — Montgomery+Shenzhen then BUSI (my recommendation: cheapest
   path to a genuine cross-modality claim), or go straight to a large corpus
   (PCam/NIH) for scale?
4. **Fine-tuning comparison arm?** Everything assumes the frozen classifier is a
   hard constraint. If a reviewer would demand a fine-tuned comparison, scope it
   now rather than retrofitting.
