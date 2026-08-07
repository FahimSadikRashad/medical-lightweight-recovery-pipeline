"""Shared preamble for the stage scripts."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robustmed import (config, corruptions, data, engine, models,  # noqa: E402,F401
                       store)


def parse_args(**extra):
    """--limit gives a fast smoke run over the first N images per split."""
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="cap each split to N images (smoke run)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None,
                   help="input resolution; 224 matches MedMNIST-C's own "
                        "severity calibration, 64 is the legacy setting")
    p.add_argument("--dataset", default=None,
                   help="corpus to run on; scopes every output path")
    p.add_argument("--registry", default=None,
                   help="MedMNIST-C corruption registry to borrow "
                        "(defaults to --dataset; set for non-MedMNIST corpora)")
    for flag, kwargs in extra.items():
        p.add_argument(f"--{flag.replace('_', '-')}", **kwargs)
    return p.parse_args()


def setup(args):
    """Seed, select the corpus, load splits, return (train, val, test, info, n)."""
    # Must happen before anything touches config paths or the corruption
    # registry, since both are cached on first use.
    if getattr(args, "image_size", None):
        config.IMAGE_SIZE = int(args.image_size)
    if (getattr(args, "dataset", None) or getattr(args, "registry", None)
            or getattr(args, "image_size", None)):
        from robustmed import corruptions
        config.set_dataset(args.dataset or config.DATA_FLAG,
                           args.registry or args.dataset)
        corruptions.reset()

    engine.set_seed()
    print("device:", engine.DEVICE)
    print(f"resolution: {config.IMAGE_SIZE}px")
    print(f"dataset: {config.DATA_FLAG}  "
          f"(corruption registry: {config.CORRUPTION_REGISTRY_FLAG})")
    print(f"outputs: {config.ROOT / config.DATA_FLAG}")
    train, val, test, info = data.load_dataset()
    n = len(info["label"])

    # TRAIN_CORRUPTIONS is chest-X-ray specific and does not exist in most
    # registries -- gaussian_blur is missing from 6 of 12, gaussian_noise from
    # 5. Re-pick one family per category from what this registry actually has,
    # so the training protocol stays comparable across modalities even though
    # the family names differ. No-op for chest X-ray.
    try:
        from robustmed import corruptions
        picked = config.train_corruptions_for(corruptions.names())
        if set(picked) != set(config.TRAIN_CORRUPTIONS):
            print(f"corruptions: {config.TRAIN_CORRUPTIONS} -> {picked} "
                  f"(matched by category to the {config.CORRUPTION_REGISTRY_FLAG} registry)")
            config.TRAIN_CORRUPTIONS = picked
    except Exception as exc:                     # registry needs ImageMagick
        print(f"note: could not check registry ({type(exc).__name__})")
    if args.limit:
        print(f"\n{'='*66}\n  SMOKE RUN -- splits capped to {args.limit} images.\n"
              "  Results are for plumbing only and are NOT comparable.\n"
              "  Recovery cannot learn from this little data before the lambda\n"
              "  ramp reaches full strength, so arms collapse to the same\n"
              "  constant output and every score converges to one number.\n"
              f"  Drop --limit (and --epochs) for a real run.\n{'='*66}")
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
