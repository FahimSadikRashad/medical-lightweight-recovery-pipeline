"""Non-MedMNIST corpora: real clinical images, native resolution, own splits.

Everything measured so far is on PneumoniaMNIST, which is a 64/224px re-release
of Kermany with a known provenance problem (PRESENTATION_FEEDBACK.md §0). A
MedMNIST sibling would not test generality -- same preprocessing, same release
pipeline. These are separate corpora with separate acquisition.

Each loader returns the same contract as data.load_medmnist(): an indexable
split yielding (PIL image, label) plus an `info` dict with a "label" mapping, so
corruption, training and evaluation code is untouched.

Two things every corpus here must declare, because both are choices rather than
facts:

  RESOLUTION   Images are resized to config.IMAGE_SIZE. MedMNIST-C calibrates
               corruption severity at 224, and severity is resolution-dependent
               -- a blur defined in pixels destroys a small image and barely
               marks a 4000px scan. Resizing to the calibration resolution is
               what keeps "severity 4" meaning the same thing here as it does on
               PneumoniaMNIST. Say so in the paper.

  REGISTRY     None of these have a MedMNIST-C registry of their own, so they
               borrow one by modality: chest X-ray -> pneumoniamnist,
               ultrasound -> breastmnist. Defensible, not free. Set via
               --registry, and config.train_corruptions_for() then re-picks one
               family per category from whatever that registry actually holds.

Splits are generated here, deterministically and stratified by label, because
none of these corpora ship an official one. Fixed seed so the split is stable
across runs and machines.
"""
import glob
import hashlib
import os
import zipfile

import numpy as np
from PIL import Image

from . import config

SPLIT_SEED = 12345
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)      # train / val / test


class _ImageFolder:
    """(PIL RGB at IMAGE_SIZE, int label) from a list of (path, label)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, lab = self.items[i]
        img = Image.open(path).convert("RGB").resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
        return img, lab


def _split(items, fractions=SPLIT_FRACTIONS, seed=SPLIT_SEED):
    """Deterministic stratified split.

    Stratified because these corpora are small and imbalanced -- an unstratified
    split of 800 images can easily put most positives in one side and make the
    val curve meaningless. Seeded so the split does not move between runs.
    """
    rng = np.random.default_rng(seed)
    by_label = {}
    for path, lab in items:
        by_label.setdefault(lab, []).append((path, lab))

    out = [[], [], []]
    for lab in sorted(by_label):
        group = sorted(by_label[lab])          # sort first: glob order is not stable
        idx = rng.permutation(len(group))
        n_tr = int(round(fractions[0] * len(group)))
        n_va = int(round(fractions[1] * len(group)))
        for j, k in enumerate(idx):
            bucket = 0 if j < n_tr else (1 if j < n_tr + n_va else 2)
            out[bucket].append(group[k])
    return out


def _unzip(zip_path, dest):
    if not os.path.isdir(dest) or not os.listdir(dest):
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    return dest


# --- Montgomery + Shenzhen TB chest X-rays ---------------------------------
# NLM, openly downloadable, no registration. Together ~800 posteroanterior chest
# radiographs with TB / normal labels, from two continents and two acquisition
# setups. Adult clinical data, against PneumoniaMNIST's pediatric source -- so
# this is a genuine domain shift, not a resolution ablation.
#
# Label convention: the filename ends _0 (normal) or _1 (abnormal), e.g.
# MCUCXR_0001_0.png / CHNCXR_0470_1.png.

def montgomery_shenzhen(root=None):
    root = root or os.environ.get("TB_CXR_DIR", "data/external/tb_cxr")
    items = []
    for pattern in ("**/MCUCXR_*.png", "**/CHNCXR_*.png",
                    "**/MCUCXR_*.jpg", "**/CHNCXR_*.jpg"):
        for path in glob.glob(os.path.join(root, pattern), recursive=True):
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem.endswith("_0"):
                items.append((path, 0))
            elif stem.endswith("_1"):
                items.append((path, 1))
    if not items:
        raise FileNotFoundError(
            f"no Montgomery/Shenzhen images under {root!r}. "
            f"Run scripts/download_external.py --corpus tb_cxr first.")

    tr, va, te = _split(items)
    n_pos = sum(l for _, l in items)
    print(f"tb_cxr: {len(items)} images ({n_pos} abnormal / {len(items)-n_pos} "
          f"normal) -> {len(tr)}/{len(va)}/{len(te)}")
    info = {"label": {"0": "normal", "1": "abnormal"},
            "task": "binary-class", "n_channels": 3}
    return _ImageFolder(tr), _ImageFolder(va), _ImageFolder(te), info


# --- BUSI breast ultrasound -------------------------------------------------
# Folders benign/ malignant/ normal/, with segmentation masks alongside the
# images as *_mask.png. The masks must be excluded or they enter as images.
# Collapsed to binary (normal vs lesion) to stay comparable with the chest
# X-ray tables, where chance is 0.50.

def busi(root=None, binary=True):
    root = root or os.environ.get("BUSI_DIR", "data/external/busi")
    classes = {"normal": 0, "benign": 1, "malignant": 1 if binary else 2}
    items = []
    for cls, lab in classes.items():
        for ext in ("png", "jpg", "jpeg"):
            for path in glob.glob(os.path.join(root, "**", cls, f"*.{ext}"),
                                  recursive=True):
                if "_mask" in os.path.basename(path).lower():
                    continue                    # segmentation mask, not an image
                items.append((path, lab))
    if not items:
        raise FileNotFoundError(
            f"no BUSI images under {root!r}. Expected benign/ malignant/ normal/ "
            f"subfolders; see scripts/download_external.py --corpus busi.")

    tr, va, te = _split(items)
    print(f"busi: {len(items)} images -> {len(tr)}/{len(va)}/{len(te)}")
    labels = ({"0": "normal", "1": "lesion"} if binary else
              {"0": "normal", "1": "benign", "2": "malignant"})
    info = {"label": labels,
            "task": "binary-class" if binary else "multi-class", "n_channels": 3}
    return _ImageFolder(tr), _ImageFolder(va), _ImageFolder(te), info


# Registered in data.EXTERNAL_LOADERS at import; --registry supplies the
# modality-matched corruption set.
LOADERS = {
    "tb_cxr": montgomery_shenzhen,      # --registry pneumoniamnist
    "busi": busi,                       # --registry breastmnist
}

SUGGESTED_REGISTRY = {
    "tb_cxr": "pneumoniamnist",
    "busi": "breastmnist",
}
