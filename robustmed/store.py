"""Save/load results as JSON so stages don't depend on each other's memory.

This is the fix for the worst thing in the original notebook: Baseline 1's
numbers were re-typed by hand into a literal dict for Day 3 to use. Now each
stage writes its numbers and later stages read them back.
"""
import json

from . import config

# canonical names -- import these instead of typing strings
BASELINE1 = "baseline1_frozen"
BASELINE2 = "baseline2_augmented"
RECOVERY_SWEEP = "recovery_sweep"
RECOVERY_COST = "recovery_cost"
RECONSTRUCTION = "reconstruction_quality"
ABLATION = "ablation"
STACKED = "recovery_plus_aug"
COMPOUND = "compound_corruptions"
TRANSFER = "kermany_transfer"
DATASET_STATS = "dataset_stats"
STABILITY = "stability"
CORRUPTION_SWEEP = "corruption_sweep"
CLASSIFIER_FREE = "classifier_free_restoration"


def save(name, payload):
    path = config.RESULT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"saved -> {path}")
    return path


def load(name):
    path = config.RESULT_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run that stage first (see docs/EXPERIMENTS.md)")
    return json.loads(path.read_text())


def load_optional(name):
    """For figures that should still render when an experiment isn't done yet."""
    path = config.RESULT_DIR / f"{name}.json"
    if not path.exists():
        print(f"note: {name} not run yet, skipping")
        return None
    return json.loads(path.read_text())


def save_table(name, df):
    path = config.RESULT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"saved -> {path}")
    return path


def available():
    return sorted(p.stem for p in config.RESULT_DIR.glob("*.json"))
