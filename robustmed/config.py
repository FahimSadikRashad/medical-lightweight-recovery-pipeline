"""All knobs in one place. Edit here, not in the scripts.

On Colab, point ROOT at Drive so a timeout never costs you a run:
    import os; os.environ["ROBUSTMED_ROOT"] = "/content/drive/MyDrive/cse6207"
"""
import os
from pathlib import Path

# --- where everything is written ------------------------------------------
ROOT = Path(os.environ.get("ROBUSTMED_ROOT", "runs"))

# --- data ------------------------------------------------------------------
SEED = 0
DATA_FLAG = os.environ.get("ROBUSTMED_DATASET", "pneumoniamnist")
# MedMNIST-C calibrated its corruption severities at 224, so running there means
# the fault model is cited rather than re-derived -- which is what makes the
# resolution-vs-severity problem (B2) disappear instead of needing an answer.
# Costs ~12x the pixels: the condition cache stops fitting in RAM and rendering
# stops being a per-run expense, hence the disk cache in engine.
IMAGE_SIZE = int(os.environ.get("ROBUSTMED_IMAGE_SIZE", 224))
BATCH_SIZE = 128
NUM_WORKERS = 2

# Which MedMNIST-C corruption registry to borrow. Split from DATA_FLAG so a
# non-MedMNIST corpus (Montgomery, Shenzhen, BUSI) can use a modality-matched
# registry -- chest X-ray corpora borrow pneumoniamnist. This is a defensible
# choice, not a free one: say so in the paper.
CORRUPTION_REGISTRY_FLAG = os.environ.get("ROBUSTMED_REGISTRY", "") or DATA_FLAG

# --- output directories, scoped per dataset --------------------------------
# Scoped so a second corpus cannot silently overwrite the first one's results --
# the failure that made the original multi-seed run look like a capacity finding.
CKPT_DIR = RESULT_DIR = FIG_DIR = None
BASELINE1_CKPT = BASELINE2_CKPT = None


def set_image_size(px):
    """Change resolution and re-point every output path. Call before set_dataset."""
    global IMAGE_SIZE
    IMAGE_SIZE = int(px)
    set_dataset(DATA_FLAG, CORRUPTION_REGISTRY_FLAG)


def set_dataset(flag, registry_flag=None):
    """Point every output path at `flag`'s subtree and rebind the checkpoints.

    Call before anything touches the filesystem. `corruptions.reset()` must
    follow if the registry changed, since the registry is cached on first use.
    """
    global DATA_FLAG, CORRUPTION_REGISTRY_FLAG
    global CKPT_DIR, RESULT_DIR, FIG_DIR, BASELINE1_CKPT, BASELINE2_CKPT

    DATA_FLAG = flag
    CORRUPTION_REGISTRY_FLAG = registry_flag or flag

    # Resolution is part of the experiment identity, not a detail: a checkpoint
    # trained at 64 is meaningless at 224 and the corruption severities mean
    # different things. Scoping the path stops the two silently mixing.
    base = ROOT / flag / f"r{IMAGE_SIZE}"
    CKPT_DIR, RESULT_DIR, FIG_DIR = (base / "checkpoints", base / "results",
                                     base / "figures")
    for _d in (ROOT, base, CKPT_DIR, RESULT_DIR, FIG_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    BASELINE1_CKPT = CKPT_DIR / "baseline1_frozen.pt"
    BASELINE2_CKPT = CKPT_DIR / "baseline2_augmented.pt"

# --- severity convention ---------------------------------------------------
# medmnistc's apply(img, severity) is 0-indexed into 5 severity levels.
# The MedMNIST-C paper numbers them 1-5. We use 0-indexed everywhere and
# add 1 only in figure labels.
N_SEVERITIES = 5

# --- classifier (Baseline 1 and 2) -----------------------------------------
CLF_EPOCHS = 10
CLF_LR = 1e-4

# --- recovery autoencoder --------------------------------------------------
AE_WIDTHS = [4, 8, 16, 32]
AE_EPOCHS = 10
AE_LR = 1e-3
AE_RESIDUAL = False
# Perceptual loss = MSE + LAMBDA * CE(frozen_clf(recon), label).
# LAMBDA ramps from 0 after WARMUP epochs. MSE alone learns the identity map
# and recovers nothing -- see docs/FINDINGS.md.
AE_LAMBDA_MAX = 1.5
AE_WARMUP = 2

# The CE term has a trivial minimiser: because the dataset is imbalanced, an AE
# that emits anything the classifier confidently labels the majority class
# scores low CE without restoring anything. At LAMBDA_MAX=1.5 that shortcut wins
# on roughly one seed in three, and because training saved the FINAL epoch the
# collapsed weights were the ones kept. These two knobs are the fix.
AE_SELECT_BEST = True         # keep the best epoch by validation balanced accuracy
VAL_PROBE_N = 256             # val images per condition in the selection probe

# How a recovery module maps its raw output back into [0,1]. This is NOT a
# refinement -- it is the difference between training and not training.
#   "clamp"    torch.clamp(y, 0, 1). ZERO gradient outside [0,1], so a decoder
#              that initialises negative everywhere is dead permanently. This
#              caused 9 of 20 stability runs to produce no usable model. Kept
#              only so the regression check can reproduce the failure.
#   "residual" clamp(x + delta). Starts in range, so the clamp rarely saturates.
#   "sigmoid"  always-nonzero gradient, no residual path.
# Every recovery module shares this, so the module comparison is not confounded
# by some architectures carrying a global residual and others not.
RECOVERY_OUTPUT = "residual"
RECOVERY_OUTPUTS = ("clamp", "residual", "sigmoid")

# A model predicting one class is collapsed, but so is one that predicts two and
# sits at chance -- s0_w4 scored bal=0.5043 and the pred_counts test missed it.
COLLAPSE_BAL = 0.52

# Stability sweep (scripts/11_stability.py)
AE_SEEDS = [0, 1, 2, 3, 4, 5]
AE_LAMBDAS = [0.0, 0.25, 0.5, 1.0, 1.5]   # 0.0 doubles as the MSE-only ablation

TRAIN_CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "jpeg_compression"]
TRAIN_SEVERITIES = [0, 1, 2]
HOLDOUT_SEVERITY = 4          # never seen while training recovery

# --- corruption taxonomy ---------------------------------------------------
# MedMNIST-C registries are modality-specific and share almost nothing by NAME:
# across the 12 registries only contrast_down, jpeg_compression and pixelate are
# universal. gaussian_blur is absent from 6 of them, gaussian_noise from 5. So
# any cross-dataset statement has to be made at the level of CATEGORY, not
# corruption name -- "blur" transfers, "gaussian_blur" does not.
#
# The categories also name the operation a recovery module would need:
#   noise        low-pass
#   blur         high-pass
#   photometric  intensity remap -- NEITHER spatial filter helps
#   codec        block/quantization artifacts
# That third row is the reason this taxonomy is in the code and not just a
# comment: 6 of the 10 families we never train on are photometric, and a module
# built from spatial convolutions has no mechanism for them at all.
CORRUPTION_CATEGORIES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "blur": ["gaussian_blur", "defocus_blur", "motion_blur", "zoom_blur"],
    "photometric": ["brightness_up", "brightness_down", "contrast_up",
                    "contrast_down", "gamma_corr_up", "gamma_corr_down",
                    "saturate"],
    "codec": ["jpeg_compression", "pixelate"],
    "artifact": ["bubble", "stain_deposit", "black_corner", "characters"],
}


def category_of(name):
    for cat, members in CORRUPTION_CATEGORIES.items():
        if name in members:
            return cat
    return "other"


def train_corruptions_for(available):
    """Pick one family per category from whatever this registry actually has.

    Keeps the training protocol comparable across modalities: the same
    categories every time, even though the family names differ. Falls back to
    TRAIN_CORRUPTIONS when they are all present (i.e. chest X-ray), so existing
    results are unaffected.
    """
    available = set(available)
    if set(TRAIN_CORRUPTIONS) <= available:
        return list(TRAIN_CORRUPTIONS)
    picked = []
    for cat in ("noise", "blur", "codec"):
        hit = [c for c in CORRUPTION_CATEGORIES[cat] if c in available]
        if hit:
            picked.append(hit[0])
    return picked

# --- evaluation ------------------------------------------------------------
EVAL_CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "jpeg_compression"]
EVAL_SEVERITIES = [0, 2, 4]   # mild / mid / strong; 4 is the unseen holdout

HEADLINE_WIDTH = 4            # the width reported in the paper's main figures

COMPOUND_PAIRS = [
    ("gaussian_noise", 2, "gaussian_blur", 2),
    ("gaussian_blur", 2, "jpeg_compression", 2),
    ("gaussian_noise", 2, "jpeg_compression", 2),
]

TRANSFER_CORRUPTION = "gaussian_noise"
TRANSFER_SEVERITY = 2
KERMANY_TEST_DIR = os.environ.get("KERMANY_TEST_DIR", "")

# --- checkpoint filenames --------------------------------------------------

def ae_ckpt(width, seed=None, tag=None, arch=None):
    """Recovery module checkpoint path.

    seed=None, tag=None and arch=None reproduce the original single-run filename,
    so existing checkpoints keep loading. Passing any of them scopes the file,
    which is what lets a sweep over arch/seed/lambda/output run without runs
    silently overwriting each other -- the failure that made the first
    multi-seed result look like a capacity finding.
    """
    parts = ["recovery_ae" if arch in (None, "convae") else f"recovery_{arch}"]
    # width=None means "the paper's own configuration", not a missing value
    parts.append("published" if width is None else f"w{width}")
    if seed is not None:
        parts.append(f"s{seed}")
    if tag:
        parts.append(tag)
    return CKPT_DIR / ("_".join(parts) + ".pt")


# Populate the paths for the default dataset at import time.
set_dataset(DATA_FLAG, CORRUPTION_REGISTRY_FLAG)
