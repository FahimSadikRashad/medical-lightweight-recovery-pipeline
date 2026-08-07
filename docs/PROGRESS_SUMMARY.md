# Progress summary

Frozen medical classifier + a tiny input-space recovery module. Question: which
module design gives corruption robustness under a compute constraint?

---

## 1. A bug invalidated the previous results, and finding it was the first result

The earlier stability sweep produced **9 dead runs out of 20**. Cause: the
module's output went through `torch.clamp(out, 0, 1)`, which has **zero gradient
outside [0,1]**. A decoder that initialises negative everywhere is dead
permanently — no gradient ever reaches the weights.

Evidence it was that and not bad luck: two runs at *different widths and
different seeds* had bit-identical loss curves, `loss = 0.3572 + λ·4.2037`, for
ten epochs. Reproduced locally — those exact seeds give 100% negative pre-clamp
output and gradient exactly `0.00e+00`.

Fixed by giving every module one shared gradient-safe output. **Failures went
9/20 → 0/24.**

Why it mattered beyond the bug: most modern restoration architectures carry a
global residual and are immune to this trap. Left in place, they would have
looked more stable than the baseline for a reason unrelated to their design.

## 2. Evaluating 3 corruption families out of 13 was hiding the real behaviour

The pipeline trained on 3 families and evaluated the same 3. Widening to the
full registry (13 families × 5 severities) changed the conclusion:

| group | mean gain |
|---|---|
| seen family, seen severity | +0.142 |
| seen family, unseen severity | +0.124 |
| **unseen family** | **+0.016** |

**Severity extrapolation works. Category extrapolation does not.** That is the
honest generalization boundary, and it is only visible with the full registry.

## 3. Benchmarked published architectures, and one mechanism explains the ordering

Five architectures at their published configurations, plus two zero-parameter
controls. Then a controlled ablation: one skeleton, one budget, one mechanism at
a time, 6 seeds, analysed **paired by seed** (seed effects are shared across
arms — pairing cuts the standard deviation from 0.081 to 0.025).

| mechanism | paired Δ mCE vs plain | p |
|---|---|---|
| **multi-scale (downsample/upsample)** | **−0.301** | **0.008** |
| spatially-adaptive modulation | −0.124 | 0.148 |
| parameter-free attention | −0.098 | 0.314 |
| channel attention | +0.154 | 0.093 |
| gating | +0.191 | 0.204 |

**Only multi-scale is significant.** It also explains the whole architecture
ranking: ConvAE won because it downsamples; SAFMN was second because multi-level
pooling is a partial multi-scale; SPAN and DnCNN, both same-resolution, were
worst.

And it is the *cheapest* mechanism — downsampling reduces compute, so at equal
MACs multi-scale buys 3× the parameters. The mechanism that works is the one
that saves compute.

## 4. Moving to 224px resolved a methodological problem and improved the result

MedMNIST-C calibrates its corruption severities at 224. Running at 64 meant
severities did not mean what the benchmark says they mean — and a fixed-pixel
blur destroys a 64px image while barely touching a large one. At 224 the fault
model is **cited rather than re-derived**.

Current pilot at 224 (frozen classifier = 2.23M params, clean 0.876, collapses
to chance under all noise severities):

| module | params | mCE ↓ | categories harmed |
|---|---|---|---|
| **ConvAE** | **14,067 (0.6% of classifier)** | **0.613** | **none** |
| fixed unsharp filter | 0 | 0.861 | codec, noise |
| identity | 0 | 1.000 | — |

Per-category gain for the 14k module: noise **+0.233**, codec **+0.215**, blur
**+0.210**, photometric **+0.035**.

Cost: **55 MMACs, 1.16 ms** on one CPU thread — the constrained-hardware figure.

At 64px, photometric corruption was the binding failure (every learned module
made it *worse* than doing nothing). At 224 it is positive. That failure was
substantially an artifact of the low resolution.

---

## Honest caveats

- The 224 result is **n=1 seed**. The 6-seed statistics are from 64px. Multi-seed
  at 224 is the immediate next run.
- **NAFNet (29M) and SPAN diverge** under this training protocol on 4,708 images
  — the CE term destabilises and best-epoch selection returns a pure-MSE model.
  Reported as a *training* result, not an architecture verdict, with the
  protocol stated. A `ce_used` column flags any row where this happened.
- No cross-dataset validation yet.

## Next

1. Multi-seed at 224 to put error bars on the pilot.
2. **K3** — the module the measurements now specify: multi-scale spatial
   recovery + a dedicated photometric branch + a learned identity fallback +
   a ~100-parameter gate.
3. Cross-modality: chest X-ray (Montgomery/Shenzhen) then ultrasound (BUSI).
   Verified that only 3 corruption families are common to all 12 MedMNIST-C
   registries, so cross-dataset claims must be made per **category**, never per
   corruption name.
