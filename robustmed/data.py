"""Datasets and loaders.

One convention everywhere: datasets yield float CHW tensors in [0, 1],
unnormalized. ImageNet normalization lives inside models.Classifier.

The original notebook normalized inside some datasets and not others, so the
same loader could not feed both the autoencoder (pixel space) and the
classifier. This is that fix.
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from . import config
from .corruptions import corrupt, corrupt_compound

CLEAN = "clean"


def to_tensor01(img):
    """PIL image or uint8 HWC array -> float CHW tensor in [0, 1].

    copy=True is required, not defensive: medmnist returns read-only arrays and
    np.ascontiguousarray passes those straight through, so torch.from_numpy
    warns about a non-writable tensor on every single sample. The copy is 12 KB
    per 64x64 image.
    """
    a = np.array(img, order="C", copy=True)
    return torch.from_numpy(a).float().permute(2, 0, 1) / 255.0


def label_of(lab):
    return int(np.asarray(lab).squeeze())


# --- condition names -------------------------------------------------------
# A condition is "clean" or "<corruption>_sev<n>". These two functions are the
# only place that format is known. The original parsed severity with c[-1],
# which silently truncates at severity 10+.

def condition(name, severity):
    return CLEAN if name is None else f"{name}_sev{severity}"


def parse_condition(cond):
    if cond == CLEAN:
        return None, None
    name, sev = cond.rsplit("_sev", 1)
    return name, int(sev)


def eval_conditions(corruptions=None, severities=None, include_clean=True):
    corruptions = corruptions or config.EVAL_CORRUPTIONS
    severities = severities or config.EVAL_SEVERITIES
    conds = [CLEAN] if include_clean else []
    return conds + [condition(c, s) for c in corruptions for s in severities]


def pretty(cond):
    """Short figure label, with severity shown in the paper's 1-indexed form."""
    name, sev = parse_condition(cond)
    if name is None:
        return "clean"
    short = name.replace("gaussian_", "g_").replace("jpeg_compression", "jpeg")
    return f"{short}\ns{sev + 1}"


# --- MedMNIST splits -------------------------------------------------------

def load_medmnist():
    """Returns (train, val, test, info). Images stay PIL for the corruption API."""
    import medmnist
    from medmnist import INFO

    info = INFO[config.DATA_FLAG]
    cls = getattr(medmnist, info["python_class"])
    splits = [cls(split=s, download=True, size=config.IMAGE_SIZE, as_rgb=True)
              for s in ("train", "val", "test")]
    return splits[0], splits[1], splits[2], info


def class_counts(raw, info):
    labels = np.array([label_of(raw[i][1]) for i in range(len(raw))])
    return {info["label"][str(k)]: int((labels == k).sum()) for k in np.unique(labels)}


def subset(raw, n):
    """First n items -- for a fast smoke run. n=None returns the full split."""
    return raw if n is None else Subset(raw, range(min(n, len(raw))))


# --- datasets --------------------------------------------------------------

class Clean(Dataset):
    """pil_transform is where AugMedMNISTC goes for Baseline 2 training."""

    def __init__(self, raw, pil_transform=None):
        self.raw, self.pil_transform = raw, pil_transform

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        img, lab = self.raw[i]
        if self.pil_transform:
            img = self.pil_transform(img)
        return to_tensor01(img), label_of(lab)


class Corrupted(Dataset):
    """One fixed (corruption, severity). Evaluation only -- fixed keeps it reproducible."""

    def __init__(self, raw, name, severity):
        self.raw, self.name, self.severity = raw, name, severity

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        img, lab = self.raw[i]
        return to_tensor01(corrupt(img, self.name, self.severity)), label_of(lab)


class Compound(Dataset):
    """Two stacked corruptions -- unseen fault mode (Sub-Q1)."""

    def __init__(self, raw, name1, sev1, name2, sev2):
        self.raw, self.spec = raw, (name1, sev1, name2, sev2)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        img, lab = self.raw[i]
        return to_tensor01(corrupt_compound(img, *self.spec)), label_of(lab)


class Pairs(Dataset):
    """(corrupted, clean, label) for recovery training.

    A random corruption+severity is drawn per item, so each epoch sees a
    different mix.
    """

    def __init__(self, raw, corruptions, severities):
        self.raw = raw
        self.corruptions = list(corruptions)
        self.severities = list(severities)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        img, lab = self.raw[i]
        name = self.corruptions[np.random.randint(len(self.corruptions))]
        sev = self.severities[np.random.randint(len(self.severities))]
        return to_tensor01(corrupt(img, name, sev)), to_tensor01(img), label_of(lab)


class Kermany(Dataset):
    """Kermany chest X-rays downsampled to match training (Sub-Q2 transfer).

    Labels mapped NORMAL=0 / PNEUMONIA=1 to match PneumoniaMNIST's ordering.
    """

    def __init__(self, items, name=None, severity=None):
        self.items, self.name, self.severity = items, name, severity

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, lab = self.items[i]
        img = Image.open(path).convert("RGB").resize((config.IMAGE_SIZE,) * 2)
        if self.name:
            img = Image.fromarray(corrupt(img, self.name, self.severity))
        return to_tensor01(img), int(lab)


def kermany_items(test_dir=None):
    test_dir = test_dir or config.KERMANY_TEST_DIR
    items = []
    for cls_name, lab in (("NORMAL", 0), ("PNEUMONIA", 1)):
        files = []
        for ext in ("jpeg", "jpg", "png"):
            files += glob.glob(os.path.join(test_dir, cls_name, f"*.{ext}"))
        print(f"{cls_name}: {len(files)} images")
        items += [(f, lab) for f in sorted(files)]
    return items


# --- loaders ---------------------------------------------------------------

def loader(dataset, shuffle=False, batch_size=None):
    g = torch.Generator()
    g.manual_seed(config.SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        worker_init_fn=_seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )


def condition_loader(raw, cond):
    """Eval loader for a condition name."""
    name, sev = parse_condition(cond)
    ds = Clean(raw) if name is None else Corrupted(raw, name, sev)
    return loader(ds)


def _seed_worker(worker_id):
    """Corruption sampling happens in workers; without this, pairs aren't reproducible."""
    import random
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
