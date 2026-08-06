"""All knobs in one place. Edit here, not in the scripts.

On Colab, point ROOT at Drive so a timeout never costs you a run:
    import os; os.environ["ROBUSTMED_ROOT"] = "/content/drive/MyDrive/cse6207"
"""
import os
from pathlib import Path

# --- where everything is written ------------------------------------------
ROOT = Path(os.environ.get("ROBUSTMED_ROOT", "runs"))
CKPT_DIR = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

for _d in (ROOT, CKPT_DIR, RESULT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- data ------------------------------------------------------------------
SEED = 0
DATA_FLAG = "pneumoniamnist"
IMAGE_SIZE = 64
BATCH_SIZE = 128
NUM_WORKERS = 2

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

# Stability sweep (scripts/11_stability.py)
AE_SEEDS = [0, 1, 2]
AE_LAMBDAS = [0.0, 0.25, 0.5, 1.0, 1.5]   # 0.0 doubles as the MSE-only ablation

TRAIN_CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "jpeg_compression"]
TRAIN_SEVERITIES = [0, 1, 2]
HOLDOUT_SEVERITY = 4          # never seen while training recovery

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
BASELINE1_CKPT = CKPT_DIR / "baseline1_frozen.pt"
BASELINE2_CKPT = CKPT_DIR / "baseline2_augmented.pt"


def ae_ckpt(width, seed=None, tag=None):
    """Recovery AE checkpoint path.

    seed=None and tag=None reproduce the original single-run filename, so
    existing checkpoints keep loading. Passing either scopes the file, which is
    what lets a sweep over seeds/lambdas/residual run without runs silently
    overwriting each other -- the failure that made the first multi-seed result
    look like a capacity finding.
    """
    parts = [f"recovery_ae_w{width}"]
    if seed is not None:
        parts.append(f"s{seed}")
    if tag:
        parts.append(tag)
    return CKPT_DIR / ("_".join(parts) + ".pt")
