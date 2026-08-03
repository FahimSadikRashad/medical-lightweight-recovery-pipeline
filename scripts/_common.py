"""Shared preamble for the stage scripts."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robustmed import config, data, engine, models, store  # noqa: E402,F401


def parse_args(**extra):
    """--limit gives a fast smoke run over the first N images per split."""
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="cap each split to N images (smoke run)")
    p.add_argument("--epochs", type=int, default=None)
    for flag, kwargs in extra.items():
        p.add_argument(f"--{flag.replace('_', '-')}", **kwargs)
    return p.parse_args()


def setup(args):
    """Seed, load splits, return (train, val, test, info, n_classes)."""
    engine.set_seed()
    print("device:", engine.DEVICE)
    train, val, test, info = data.load_medmnist()
    n = len(info["label"])
    if args.limit:
        print(f"smoke run: capping splits to {args.limit} images")
        train, val, test = (data.subset(s, args.limit) for s in (train, val, test))
    return train, val, test, info, n


def frozen_baseline1(n_classes):
    """Load Baseline 1 and freeze it. Every later stage uses this exact model."""
    clf = models.Classifier(n_classes).to(engine.DEVICE)
    if not config.BASELINE1_CKPT.exists():
        raise SystemExit("run scripts/01_baseline1_frozen.py first")
    engine.load_ckpt(config.BASELINE1_CKPT, clf)
    return clf.freeze()


def balanced(payload):
    """{condition: metrics} -> {condition: balanced_accuracy}."""
    return {k: v["balanced_accuracy"] for k, v in payload.items()}
