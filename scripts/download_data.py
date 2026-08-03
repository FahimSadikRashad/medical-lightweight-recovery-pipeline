"""Download both datasets and report where they landed.

    python scripts/download_data.py              # PneumoniaMNIST only
    python scripts/download_data.py --kermany    # + Kermany (needs Kaggle access)

PneumoniaMNIST caches to ~/.medmnist. Kermany comes from Kaggle and is only
needed for the cross-domain transfer stage.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robustmed import config, data  # noqa: E402


def download_medmnist():
    print(f"== {config.DATA_FLAG} at {config.IMAGE_SIZE}x{config.IMAGE_SIZE} ==")
    train, val, test, info = data.load_medmnist()
    print("task:", info["task"])
    print("labels:", info["label"])
    print("sizes: train=%d val=%d test=%d" % (len(train), len(val), len(test)))
    # Class imbalance matters for how the results are read -- note it here.
    for name, split in (("train", train), ("val", val), ("test", test)):
        print(f"  {name}: {data.class_counts(split, info)}")
    return train, val, test, info


def download_kermany():
    """Kermany chest X-rays (Kaggle). Sets KERMANY_TEST_DIR for you to export."""
    print("== Kermany chest X-ray ==")
    try:
        import kagglehub
    except ImportError:
        print("pip install kagglehub, then re-run")
        return None

    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    test_dir = os.path.join(path, "chest_xray", "test")
    if not os.path.isdir(test_dir):
        print(f"downloaded to {path} but no chest_xray/test inside -- check the layout")
        return None
    print("test split:", test_dir)
    print("\nAdd this to your environment before running the transfer stage:")
    print(f'  export KERMANY_TEST_DIR="{test_dir}"')
    return test_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kermany", action="store_true",
                   help="also fetch the Kermany transfer dataset")
    args = p.parse_args()

    download_medmnist()
    if args.kermany:
        print()
        download_kermany()
    print(f"\noutputs will be written under {config.ROOT}")
