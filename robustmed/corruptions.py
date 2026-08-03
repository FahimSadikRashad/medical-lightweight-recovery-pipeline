"""Wrapper over the official MedMNIST-C corruption registry.

We use the paper's registry (Di Salvo et al., 2024) rather than hand-rolled
noise/blur, so the fault model is the benchmark's.

Needs ImageMagick + wand -- run `robustmed.colab.install()` on a fresh Colab.
"""
import numpy as np
from PIL import Image

from .config import DATA_FLAG

_registry = None


def registry():
    """Corruption objects for the dataset, keyed by name. Loaded once."""
    global _registry
    if _registry is None:
        from medmnistc.corruptions.registry import CORRUPTIONS_DS
        _registry = CORRUPTIONS_DS[DATA_FLAG]
    return _registry


def names():
    return list(registry().keys())


def corrupt(img, name, severity):
    """One corruption at a fixed severity -> uint8 HxWx3 array.

    severity is 0-indexed: 0 = mildest, 4 = strongest.
    """
    out = registry()[name].apply(img.copy(), severity)
    return np.asarray(out).astype("uint8")


def corrupt_compound(img, name1, sev1, name2, sev2):
    """Two corruptions in sequence -- a fault mode the AE never trained on.

    Not commutative; COMPOUND_PAIRS in config fixes one order per pair.
    """
    step1 = corrupt(img, name1, sev1)
    return corrupt(Image.fromarray(step1), name2, sev2)


def augmenter():
    """AugMedMNISTC: samples a random corruption AND severity per call.

    Training-time counterpart to corrupt():
      corrupt()   -> fixed severity, reproducible test sets
      augmenter() -> random per sample, this is Paper A's method
    """
    from medmnistc.augmentation import AugMedMNISTC
    return AugMedMNISTC(train_corruptions=registry())
