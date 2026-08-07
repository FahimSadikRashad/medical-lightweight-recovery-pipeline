# Reading list — what to review before running, and why

Ordered by *when it affects you*, not by topic. Tier 1 can change your
experimental design or your novelty claim, so read it before burning GPU hours.

---

## ⚠️ Read this section first: the novelty risk

**Your core mechanism has substantial prior art.** "Train a restoration network
using a loss from a frozen downstream recognition network, instead of pixel
fidelity" is an established line of work — **task-driven / recognition-aware
image restoration** — going back to at least 2018. Representative:

| Paper | Relevance |
|---|---|
| [Class-Aware Fully-Convolutional Gaussian and Poisson Denoising (arXiv:1808.06562)](https://arxiv.org/pdf/1808.06562) | 2018. Class-aware denoising — the early version of your idea. |
| [Beyond Image Super-Resolution for Image Recognition with Task-Driven Perceptual Loss (arXiv:2404.01692)](https://arxiv.org/html/2404.01692v2) | **Closest to your loss.** "Task-driven perceptual loss" guides restoration of task-relevant content. Also documents the task-network-bias problem you should expect. |
| [UniRestore: Unified Perceptual and Task-Oriented Image Restoration (arXiv:2501.13134)](https://arxiv.org/pdf/2501.13134) | Recent framing of perceptual-vs-task-oriented restoration. |
| [TaskTok: Task Tokens for Task-driven Image Restoration (arXiv:2606.26615)](https://arxiv.org/html/2606.26615v1) | Makes the point that *indiscriminate* restoration distorts semantic cues — relevant to your clean-input regression. |
| [TAPE: Task-Agnostic Prior Embedding for Image Restoration (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780438.pdf) | The task-agnostic counterpoint; relevant to your Tier-2 classifier-agnostic plan. |
| URIE, DDP, SFDUnet | Enhancement-plus-recognition pipelines; DDP aligns features between low- and high-quality images specifically to improve classification. |

**What this does and does not mean.** It does *not* invalidate your project. It
does mean the contribution cannot be "we introduce classifier-guided recovery."
Reposition to what is actually yours:

1. **The compute-bounded regime.** These methods are full-size restoration
   networks. Yours is **1,119 parameters**. Asking "how light can this be?" at
   three orders of magnitude smaller is a real and unexplored question.
2. **The capacity-vs-gain finding.** "Smallest wins; capacity does not help" is a
   genuinely novel empirical claim — nobody in the list above swept down to ~1k
   params. *This should be your headline*, and it is exactly why the multi-seed
   error bars matter so much.
3. **Frozen classifier as a hard constraint**, not a convenience — regulatory
   deployments, third-party models. Most prior work fine-tunes something.
4. **Medical imaging + the MedMNIST-C benchmark**, where this has not been done.

Do a proper search on "task-driven image restoration" and "recognition-aware
restoration" before writing the related-work section. Finding this yourself is far
better than a reviewer finding it.

---

## Tier 1 — before you run (affects design or claims)

| Paper | Why now |
|---|---|
| [**MedMNIST-C** (arXiv:2406.17536)](https://arxiv.org/abs/2406.17536) | Your Paper A and your corruption source. Read their **exact evaluation protocol and metrics** so your numbers are comparable. Note they use 224×224; you use 64×64. |
| [**MedMNIST v2** (Scientific Data)](https://www.nature.com/articles/s41597-022-01721-8) | Dataset provenance and splits. This is the paper that reveals PneumoniaMNIST comes from Kermany — the issue in `PRESENTATION_FEEDBACK.md` §0. Read the splits table carefully before adding any external dataset. |
| **Hendrycks & Dietterich, ICLR 2019 — "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations" (ImageNet-C)** | The origin of the whole 5-severity protocol MedMNIST-C inherits, and of **mCE / relative mCE**. A robustness reviewer will expect a corruption-error metric alongside balanced accuracy. Decide *now* whether to report mCE, because it changes what you log. |
| **Kermany et al., Cell 2018** | Source of PneumoniaMNIST. Read the split construction to confirm the §0 contamination finding in the original authors' own words. |
| The task-driven restoration papers above | Novelty positioning. Cheaper to know before you run than after you write. |

---

## Tier 2 — before you write (positioning and baselines)

### Your claimed lineage — read these properly, you cite them

| Paper | Why |
|---|---|
| [**Gao et al., CVPR 2023 — "Back to the Source: Diffusion-Driven Adaptation to Test-Time Corruption" (DDA)**](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_Back_to_the_Source_Diffusion-Driven_Adaptation_To_Test-Time_Corruption_CVPR_2023_paper.pdf) | **The paper your framing descends from.** They adapt *inputs* rather than the model, keep classification and generation models frozen, and argue input adaptation beats model adaptation with little/dependent/mixed data. Their argument is your argument at a different scale — know it precisely. |
| [**Oh et al., ECCV 2024 — "Efficient Diffusion-Driven Corruption Editor for Test-Time Adaptation" (Decorruptor)** (arXiv:2403.10911)](https://arxiv.org/abs/2403.10911) | **Your real competitor, and read it carefully.** They already made the *efficiency* argument here: a distilled variant at 4 NFEs, **100× faster than a diffusion baseline**. So "diffusion recovery is too slow" is a claim they have partly answered. Your differentiator must be the *magnitude* — ~1k parameters vs. a distilled latent diffusion model — quoted against their actual numbers, not against unoptimized diffusion. |
| **Nie et al., ICML 2022 — DiffPure** | Diffusion purification. Same input-adaptation mechanism, adversarial framing. Useful for showing the mechanism family. |

### Test-time adaptation — the competitors a reviewer will name

| Paper | Why |
|---|---|
| **Wang et al., ICLR 2021 — TENT** | Entropy minimization at test time. Updates BatchNorm parameters, so it **modifies the model** — which your frozen constraint forbids. Articulate that distinction explicitly; it is one of your cleanest defenses. |
| **Zhang et al., NeurIPS 2022 — MEMO** | Single-image test-time adaptation via marginal entropy. Same point: it touches the model. |

### The loss you use

| Paper | Why |
|---|---|
| **Johnson et al., ECCV 2016 — Perceptual Losses for Real-Time Style Transfer and Super-Resolution** | Origin of feature-space perceptual loss. Cite for your Tier-2 classifier-agnostic variant. |
| **Zhang et al., CVPR 2018 — LPIPS** | Supports your "PSNR/SSIM rank models differently from downstream accuracy" observation, which you have already seen empirically. |

---

## Tier 3 — when implementing baselines

### First, a design point that matters more than the paper list

Slide 9's list (DnCNN 2017, HINet 2021, NAFNet 2022) is dated, but the deeper
problem is **scale mismatch**: every one of these is millions of parameters
against your 1,119. "Our 1k module vs. their 30M model" is not an informative
comparison in either direction — you lose on quality and win on cost, trivially.

Your research question is *"how light can recovery be?"* The experiment that
actually answers it is an **architecture comparison at matched parameter
budget** — take modern restoration blocks, scale each to ~1k parameters, and ask
which architecture uses that budget best. That is a much stronger table than
comparing against published model sizes, and it is the natural experiment for
your RQ. Nobody in the literature below has run it.

So structure the baseline table in three tiers:

| Tier | What | Purpose |
|---|---|---|
| 0 params | box filter (have it) | is learning needed at all |
| **~1k params** | **your AE vs. scaled-down DnCNN / NAFNet-block / Mamba-block** | **the real comparison — which architecture wins the budget** |
| full size | one modern model at published cost | upper anchor for quality *and* cost |

### Current references (2023–2026)

| Paper | Why |
|---|---|
| [**NTIRE 2025 Image Denoising Challenge Report** (arXiv:2504.12276)](https://arxiv.org/pdf/2504.12276) | **Read this first.** Challenge reports are the fastest way to see the current frontier and the quality/efficiency trade-off across many methods at once, with consistent evaluation. Better value than any single paper. |
| [**MambaIR** (ECCV 2024)](https://link.springer.com/chapter/10.1007/978-3-031-72649-1_13) | State-space model for restoration — global receptive field at *linear* complexity, which is exactly the efficiency argument you care about. Has become a standard modern baseline. |
| **MambaIRv2** (2025) | U-shaped successor with attentive state-space and semantic-guided neighboring modules. |
| [**VAMamba** (arXiv:2509.23601)](https://arxiv.org/pdf/2509.23601) | Efficient visual adaptive Mamba for restoration. |
| **Q-MambaIR** (2025) | Quantized MambaIR — relevant if you push the constrained-hardware angle toward deployment. |
| [**PromptIR** (NeurIPS 2023)](https://arxiv.org/abs/2306.13090) | All-in-one blind restoration. Relevant because your module is also corruption-agnostic — it handles multiple corruptions without being told which. Cite it for that framing. |
| [**DyNet** (arXiv:2404.02154)](https://arxiv.org/html/2404.02154v2) | Efficient scalable all-in-one restoration. DyNet-S beats PromptIR by 0.43 dB with **31% fewer params and 57% fewer GFLOPs** — a concrete, quotable efficiency datapoint. |
| [**DaAIR** (arXiv:2405.15475)](https://arxiv.org/html/2405.15475) | Degradation-aware all-in-one restoration, explicitly smaller and more efficient than prior work. |
| [**DCPT** (arXiv:2501.15510)](https://arxiv.org/pdf/2501.15510) | Degradation-classification pre-training. Related to your corruption-gating future work. |

### Keep the old ones, but for the right reason

**DnCNN** (Zhang et al., TIP 2017) and **FFDNet** (TIP 2018) are still worth
including — not as SOTA, but as the *conventional reference point* reviewers
expect, and because DnCNN's plain residual-CNN structure is the easiest thing to
scale down to a 1k-parameter budget for the matched comparison above. **NAFNet**
(ECCV 2022) and **HINet** (CVPRW 2021) you named on slide 9, so address them even
if only to say they were superseded.

### For the external datasets

**Wang et al., CVPR 2017** (NIH ChestX-ray14) and **Irvin et al., AAAI 2019**
(CheXpert) — read the label-generation methodology. Both use NLP-mined labels,
which bounds what any transfer result on them can mean.

---

## Minimum viable reading

If time is short, these five, in order:

1. **MedMNIST-C** — your benchmark and Paper A.
2. **Task-Driven Perceptual Loss (arXiv:2404.01692)** — the novelty check.
3. **DDA (Gao, CVPR 2023)** — your framing's source.
4. **Decorruptor (ECCV 2024)** — your efficiency competitor.
5. **Hendrycks & Dietterich (ICLR 2019)** — the metric convention.

1 and 2 before running. 3 and 4 before writing a word of positioning. 5 before
you finalize what the scripts log.

---

## Claim → citation map

Fill this in as you read; it prevents unsupported sentences later.

| Your claim | Needs |
|---|---|
| Corruption causes silent collapse | Hendrycks & Dietterich; MedMNIST-C |
| Balanced accuracy is the right metric | imbalance is documented in MedMNIST v2 |
| Augmentation is the standard fix (Baseline 2) | MedMNIST-C |
| Input adaptation, not model adaptation | DDA; contrast with TENT/MEMO |
| Heavy recovery is costly | Decorruptor's own numbers — **not** a strawman |
| Classifier-guided restoration | task-driven restoration line — **cite, don't claim** |
| ~1k params is enough | **yours** — no prior work at this scale |
| Capacity does not help | **yours** — needs the multi-seed error bars |
| Recovery is classifier-coupled | task-network bias, discussed in arXiv:2404.01692 |
