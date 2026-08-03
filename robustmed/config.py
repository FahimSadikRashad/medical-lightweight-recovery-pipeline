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


def ae_ckpt(width):
    return CKPT_DIR / f"recovery_ae_w{width}.pt"
