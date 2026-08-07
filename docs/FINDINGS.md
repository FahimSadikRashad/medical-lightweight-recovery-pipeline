# Findings

## Confirmed baselines (restructured code, seeded, 64×64)

Stages 01–02 rerun. These supersede the corresponding numbers in the
"preliminary" section below and are safe to quote.

Balanced accuracy, PneumoniaMNIST test (624 images, 62.5% pneumonia):

| Condition | B1 frozen | B2 augmented |
|---|---|---|
| clean | 0.790 | 0.880 |
| gaussian_noise sev 1 / 3 / 5 | 0.500 / 0.500 / 0.500 | 0.833 / 0.804 / 0.777 |
| gaussian_blur sev 1 / 3 / 5 | 0.660 / 0.500 / 0.500 | 0.876 / 0.832 / 0.756 |
| jpeg sev 1 / 3 / 5 | 0.733 / 0.622 / 0.559 | 0.888 / 0.844 / 0.820 |
| **mean over 9 corrupted** | **0.564** | **0.826** |

- B1 collapses to a single class on 5 of 9 conditions. Raw accuracy on every
  collapsed condition is exactly 0.6250 = 390/624, the majority-class rate —
  the cleanest possible demonstration that raw accuracy hides the failure.
- **Corruption families fail qualitatively differently.** Noise collapses at the
  *mildest* severity with no graceful degradation; JPEG never collapses even at
  maximum; blur sits between. These are not one phenomenon at three magnitudes.
- B2 eliminates collapse entirely (worst case 0.756) and **also improves clean
  accuracy by +0.090** — augmentation is acting as a regularizer, so there is no
  clean-accuracy penalty to trade against it.
- **B2's corrupted mean (0.826) exceeds B1's clean accuracy (0.790).**
- Val balanced accuracy was 0.95–0.98 for both, against clean *test* of
  0.790/0.880. PneumoniaMNIST's test split is distributionally different from
  its val split. Never quote val numbers; expect a reviewer to ask.

Framing consequence: recovery must be compared against **0.564** (what is
achievable with the classifier frozen), not against 0.826. State the constraint
before the table, or the work reads as losing.

---

> **The numbers below are from the original unseeded Colab run**, carried over
> from the notebook's progress summary so the reasoning isn't lost. The
> restructured code seeds every RNG, so a rerun will shift them. Refresh this
> file from `results/` after the seeded rerun (see `EXPERIMENTS.md`).

## Setup

Recovery = convolutional autoencoder in front of a **frozen** MobileNetV2,
trained with

```
loss = MSE(recon, clean) + lambda * CE(frozen_clf(recon), label)
```

on single corruptions at severities 0–2, with severity 4 held out. Metric is
balanced accuracy throughout.

## Results

**Baseline 1 collapses.** Clean 0.830 → `gaussian_noise` 0.500 at every
severity tested, i.e. exactly chance. The collapse is *silent*: the model
predicts a single class, so raw accuracy stays near the majority-class rate
(~0.74) while balanced accuracy sits at 0.50. Any paper reporting raw accuracy
here would miss it entirely.

**Baseline 2 (augmentation, Paper A) recovers strongly** — e.g. ~0.78 on
`gaussian_noise` severity 2 — but requires retraining the classifier, which is
the thing our setting forbids.

**A ~1k-parameter AE lifts the frozen classifier from 0.50 to ~0.74** across
noise/blur/JPEG, *including the held-out severity 4*.

**Learned beats non-learned.** AE ~0.74 vs 3×3 box denoiser ~0.50–0.65. This is
what makes the result a claim about learning rather than about smoothing.

**Main RQ — gain does not increase with capacity.** w=4 gives the best mean
recovery gain; w=8/16/32 do not improve on it. Lightweight recovery isn't merely
feasible here, it's optimal. *This is the load-bearing claim and it has no error
bars yet.*

**Sub-Q1 — compound corruptions:** positive gain on all three stacked pairs
(+0.16 to +0.24), despite being a fault mode absent from training and from the
registry.

**Sub-Q2 — Kermany transfer:** recovery gain on real-world X-rays matches the
in-domain gain. The strategy survives a domain change.

## What failed

**MSE-only training.** Converges to the identity map — 0.50 on noise, i.e. no
recovery at all. The perceptual (CE) term is not a refinement, it's the
mechanism. Worth a table row of its own.

## Limitations

1. **Recovery is classifier-coupled.** An AE trained against classifier A does
   not transfer to classifier B — B collapses on A's recovered images. The
   module learns something tuned to its guidance classifier's decision surface,
   not a universal restoration. Currently an informal observation; needs the
   proper experiment in `EXPERIMENTS.md`.
2. **On clean input the AE mildly reduces accuracy.** It "fixes" images that
   aren't broken. Motivates a corruption-gating detector, and possibly the
   untested `residual=True` variant, which by construction should preserve clean
   inputs better.

## Framing

The contribution is **not** "beats augmentation" — it doesn't. It is:
retraining-free, corruption-agnostic, compute-bounded recovery for a **fixed**
classifier. A different mechanism from augmentation, useful precisely where the
classifier cannot be touched — regulatory-frozen deployments, third-party
models, on-device inference.

Lineage: test-time recovery (Gao et al., CVPR 2023; Decorruptor, ECCV 2024),
instantiated at ~1k parameters instead of a diffusion model.

The recurring ~0.73 plateau across single corruptions, compound corruptions, and
a different dataset suggests the module learns a generic projection toward the
clean manifold rather than corruption-specific fixes. That's the interesting
reading; it is not yet established over the alternative (an artifact of the
frozen classifier's decision boundary). See open question 2 in `EXPERIMENTS.md`.
