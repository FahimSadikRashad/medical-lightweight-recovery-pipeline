# Experiment ledger

Start here when resuming. Every stage writes a JSON file to
`$ROBUSTMED_ROOT/results/`; `store.available()` lists what's present.

## Done (ported from the Colab run — rerun to regenerate results/)

| # | Stage | Script | Result file |
|---|-------|--------|-------------|
| 1 | Baseline 1: clean-trained frozen classifier, collapse table | `01_baseline1_frozen.py` | `baseline1_frozen` |
| 2 | Baseline 2: MedMNIST-C augmentation (Paper A) | `02_baseline2_augmented.py` | `baseline2_augmented` |
| 3 | Recovery AE capacity sweep (w=4,8,16,32) + params/latency | `03_train_recovery.py` | `recovery_sweep`*, `recovery_cost`, `reconstruction_quality` |
| 4 | Main RQ: gain vs capacity, incl. held-out severity | `04_evaluate_recovery.py` | `recovery_sweep` |
| 5 | Ablation vs box denoiser; recovery + augmentation stacked | `05_ablation.py` | `ablation`, `recovery_plus_aug` |
| 6 | Sub-Q1 compound corruptions; Sub-Q2 Kermany transfer | `06_generalization.py` | `compound_corruptions`, `kermany_transfer` |
| 7 | Figures 1–5 | `07_figures.py` | — |
| 16 | RQ-5, classifier-free recovery: PSNR/SSIM + downstream accuracy, no CE term, all-in-one training | `16_classifier_free_restoration.py` | `classifier_free_restoration` |

\* written by stage 4, not 3.

**Stage 16 needs `13_module_comparison.py --published` run first** for the
H-M2 comparison column (classifier-free mCE vs. the CE-guided mCE for the same
architecture at the same published size) — otherwise it reports restoration
quality alone. `--arms moceir` requires `--batch-size 4` or smaller; see
[`robustmed/moceir.py`](../robustmed/moceir.py)'s deviations list for why
(7.7GB measured at batch=2, 224px, CPU, backward pass — the project's default
batch of 128 will exhaust memory).

**The numbers in `FINDINGS.md` came from the original unseeded Colab run.** The
restructured code seeds every RNG, so a rerun will not reproduce them digit for
digit. Rerun stages 1–6 and refresh `FINDINGS.md` before drafting.

## Remaining

See also [`PRESENTATION_FEEDBACK.md`](PRESENTATION_FEEDBACK.md), which turns the
progress-presentation remarks into a prioritized plan and supersedes the ordering
below where the two overlap. It also records a **blocking correction**: the
Kermany "cross-domain transfer" experiment is not cross-domain — PneumoniaMNIST
is derived from Kermany and its test split *is* Kermany's `test/` folder.

Nothing here is started. Ordered by how much the paper needs it.

### Blocking for submission

- [ ] **Seeded rerun of stages 1–6.** Current numbers predate seeding. Needed
      before any table is drafted.
- [ ] **Multiple seeds (≥3) with mean ± std.** Single-run numbers on a test
      split this small won't survive review. The main claim — that w=4 beats
      larger widths — needs error bars, because the gaps between widths may be
      inside run-to-run noise. Highest-value remaining experiment.
- [ ] **Full corruption set, not 3 families.** Evaluation covers
      `gaussian_noise`, `gaussian_blur`, `jpeg_compression`; the PneumoniaMNIST
      registry has more (`corruptions.names()`). Reviewers will ask whether the
      three were picked post hoc. Widen `EVAL_CORRUPTIONS` to the full registry.
- [ ] **All 5 severities.** Currently 0/2/4. Cheap to add and turns three points
      into a degradation curve.

### Strengthens the contribution

- [ ] **MSE-only ablation as a reported row.** `AE_LAMBDA_MAX = 0`. Was observed
      informally and is a genuine finding — the perceptual term is what makes
      the module work at all — but it isn't currently in any table.
- [ ] **Lambda sweep.** `AE_LAMBDA_MAX ∈ {0, 0.5, 1.5, 3}`. Shows the finding
      isn't a single lucky hyperparameter.
- [ ] **`residual=True` variant.** Config flag exists, untested. Predicting
      input + correction should reduce the clean-input regression (Limitation 2)
      — that's a concrete fix, not just a caveat.
- [ ] **Second architecture.** Train Baseline 1 as ResNet18 and repeat. Tests
      whether the approach is MobileNetV2-specific. Needs a small change in
      `models.backbone` to accept an arch argument.
- [ ] **Corruption-gating detector.** Apply recovery only when corruption is
      detected. Directly addresses Limitation 2 and is the obvious extension a
      reviewer will propose.
- [ ] **Classifier-coupling experiment, properly run.** The observation that an
      AE trained against classifier A fails on classifier B is currently a
      side-note from an ad-hoc cell. Make it a table: AE trained on A/B/both,
      evaluated on A and B.

### Nice to have

- [ ] Cross-dataset check on a second MedMNIST task to test task-generality.
- [ ] Deeper AE or a U-Net skip variant, to confirm the capacity plateau isn't
      an architecture-family artifact.
- [ ] Kermany transfer across more than one corruption (currently only
      `gaussian_noise` at severity 2).
- [ ] Per-class breakdown — which class the collapse falls onto, and whether
      recovery restores both classes or just the minority one.

## Open questions to resolve before writing

1. **Is `HEADLINE_WIDTH = 4` real or noise?** The whole main-RQ framing
   ("smallest is best") rests on it. Answer with the multi-seed run.
2. **Does the recurring ~0.73 plateau across corruptions, compound corruptions,
   and Kermany mean a generic projection toward the clean manifold, or is it an
   artifact of the frozen classifier's decision boundary?** Currently
   interpreted as the former in `FINDINGS.md` — that's the more interesting
   claim and the less safe one. Probing classifier features on recovered vs.
   clean images would settle it.
3. ~~**Kermany label ordering.**~~ **Resolved.** `medmnist/info.py` gives
   PneumoniaMNIST `{"0": "normal", "1": "pneumonia"}`, so `data.Kermany`'s
   NORMAL=0 / PNEUMONIA=1 mapping is correct. (But see
   [`PRESENTATION_FEEDBACK.md`](PRESENTATION_FEEDBACK.md) §0 — the larger problem
   is that the Kermany set is not external data at all.)
