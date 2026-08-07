"""Stage 1 -- characterize EVERY corruption in the registry, not just three.

Evaluation only. No training happens here; it reuses Baseline 1 and whatever
recovery checkpoints already exist, so it costs minutes and can invalidate the
premise of the training that follows it.

Why this runs before the module comparison. We train recovery on 3 corruption
families at severities 0-2 and evaluate the SAME 3 at 0/2/4. Every claim about
which condition is hardest is therefore drawn from 3 of the registry's families.
"Blur binds" -- the finding that decides which architectures are contenders --
has never been tested against the rest. If a different family binds harder, the
arm list for Stage 2 is wrong before it runs.

Results are split three ways, because pooling them hides the interesting group:

  seen family, seen severity     in-distribution
  seen family, unseen severity   the existing holdout
  UNSEEN family entirely         the real robustness test, and the honest
                                 answer to "compare against the full set of
                                 corruptions" from the progress presentation

Conditions the baseline cannot handle are NOT excluded: that is where the frozen
classifier fails and recovery has the most to prove. Only conditions that no
model clears -- knowable after every arm has run -- are split into their own
group, so the worst case is not defined by something nobody can fix.

    python scripts/12_corruption_sweep.py
    python scripts/12_corruption_sweep.py --widths 4 16 --severities 0 1 2 3 4
    python scripts/12_corruption_sweep.py --dataset breastmnist
"""
import numpy as np
import pandas as pd

from _common import (balanced, config, corruptions, data, engine, models,
                     parse_args, setup, store, frozen_baseline1)


def group_of(name, severity, seen_families, seen_severities):
    if name not in seen_families:
        return "unseen_family"
    return "seen_sev" if severity in seen_severities else "unseen_sev"


if __name__ == "__main__":
    args = parse_args(
        widths={"type": int, "nargs": "+", "default": [config.HEADLINE_WIDTH]},
        severities={"type": int, "nargs": "+",
                    "default": list(range(config.N_SEVERITIES))},
        seeds={"type": int, "nargs": "+", "default": [None]},
        tag={"default": None},
        arch={"default": "convae"},
    )
    train, val, test, info, n_classes = setup(args)
    clf = frozen_baseline1(n_classes)

    families = corruptions.names()
    seen = set(config.TRAIN_CORRUPTIONS)
    print(f"\nregistry '{config.CORRUPTION_REGISTRY_FLAG}': {len(families)} families")
    print(f"  trained on : {sorted(seen)}")
    print(f"  never seen : {sorted(set(families) - seen)}")
    if not seen <= set(families):
        raise SystemExit(
            f"TRAIN_CORRUPTIONS {sorted(seen - set(families))} are not in this "
            f"registry. Set config.CORRUPTION_REGISTRY_FLAG to a modality match.")

    conds = [data.condition(n, s) for n in families for s in args.severities]
    print(f"  grid       : {len(families)} x {len(args.severities)} = "
          f"{len(conds)} conditions (+ clean)")

    # --- baseline over the full grid ---------------------------------------
    print("\n== baseline 1 (no recovery) ==")
    b1_raw = engine.eval_conditions(None, clf, test, [data.CLEAN] + conds)
    b1 = balanced(b1_raw)

    at_chance = [c for c in conds if b1.get(c, 1.0) <= config.COLLAPSE_BAL]
    print(f"\nbaseline at chance on {len(at_chance)}/{len(conds)} conditions.")
    print("  This is HEADROOM, not a reason to exclude them -- it is where the\n"
          "  frozen classifier fails and recovery has the most to prove.")

    # --- each recovery checkpoint over the same grid ------------------------
    rows = []
    raw = {"baseline1": b1_raw}
    for width in args.widths:
        for seed in args.seeds:
            try:
                ae = engine.load_recovery(width, seed=seed, tag=args.tag,
                                          arch=args.arch)
            except FileNotFoundError:
                print(f"\nskip w={width} seed={seed}: no checkpoint")
                continue
            key = f"{args.arch}_w{width}" + (f"_s{seed}" if seed is not None else "")
            print(f"\n== {key} ==")
            res = engine.eval_conditions(ae, clf, test, [data.CLEAN] + conds)
            raw[key] = res
            bal = balanced(res)

            for c in conds:
                name, sev = data.parse_condition(c)
                rows.append({
                    "model": key, "arch": args.arch, "width": width, "seed": seed,
                    "family": name, "severity": sev,
                    "category": config.category_of(name),
                    "group": group_of(name, sev, seen, config.TRAIN_SEVERITIES),
                    "baseline_at_chance": c in at_chance,
                    "bal": bal[c], "baseline_bal": b1[c],
                    "gain": bal[c] - b1[c],
                })

    if not rows:
        raise SystemExit("no recovery checkpoints found -- run stage 3 or 11 first")

    df = pd.DataFrame(rows)
    store.save(store.CORRUPTION_SWEEP, raw)
    store.save_table("corruption_sweep", df)

    # --- what actually binds ------------------------------------------------
    # Unfixable = no model cleared chance. Only knowable now that every model
    # has run; reported as its own group rather than pre-excluded.
    unfixable = set(engine.unfixable_conditions(
        [balanced(v) for k, v in raw.items() if k != "baseline1"], conds))
    df["unfixable"] = [data.condition(r.family, r.severity) in unfixable
                       for r in df.itertuples()]
    print(f"\nunfixable by every model: {len(unfixable)}/{len(conds)}")
    for c in sorted(unfixable):
        print(f"  {c}")
    ok = df[~df["unfixable"]]

    print("\n=== worst condition per model (excluding unfixable) ===")
    for key, g in ok.groupby("model"):
        w = g.loc[g["bal"].idxmin()]
        print(f"  {key:22s} worst={w['bal']:.4f} on {w['family']}_sev{w['severity']} "
              f"(gain {w['gain']:+.3f}, group={w['group']})")

    print("\n=== by category (the operation a module would need) ===")
    cat = ok.groupby("category").agg(
        n=("bal", "size"), families=("family", "nunique"),
        worst=("bal", "min"), mean_bal=("bal", "mean"),
        mean_gain=("gain", "mean")).reset_index().sort_values("mean_bal")
    store.save_table("corruption_by_category", cat)
    print(cat.to_string(index=False))
    print("  photometric = global intensity remap. Spatial convolution has no\n"
          "  mechanism for it, and we never train on it -- if it is the worst\n"
          "  category, K2 is a THREE-way conflict, not noise-vs-blur.")

    print("\n=== by group ===")
    grp = ok.groupby("group").agg(n=("bal", "size"), worst=("bal", "min"),
                                  mean_bal=("bal", "mean"),
                                  mean_gain=("gain", "mean")).reset_index()
    print(grp.to_string(index=False))

    print("\n=== per family: severity slope (does recovery hold as it gets worse?) ===")
    # Slope over the widest severity range each family still HAS. Anchoring on
    # the global min/max returns NaN for any family whose hardest severity was
    # dropped as unfixable -- which silently hid brightness_down, contrast_down
    # and speckle_noise, three of the families the result is about.
    fam = []
    for (name,), g in ok.groupby(["family"]):
        sevs = sorted(g["severity"].unique())
        if len(sevs) < 2:
            continue
        lo, hi = sevs[0], sevs[-1]
        a = g[g["severity"] == lo]["bal"].mean()
        b = g[g["severity"] == hi]["bal"].mean()
        fam.append({"family": name, "seen": name in seen,
                    "category": config.category_of(name),
                    "sev_lo": lo, "sev_hi": hi, "bal_lo": a, "bal_hi": b,
                    "slope": b - a, "mean_gain": g["gain"].mean()})
    fam = pd.DataFrame(fam).sort_values("slope")
    store.save_table("corruption_family_slopes", fam)
    print(fam.to_string(index=False))

    print("\n=== read this ===")
    binding = fam.iloc[0]
    print(f"  binding family : {binding['family']} (slope {binding['slope']:+.3f}, "
          f"{'trained on' if binding['seen'] else 'NEVER TRAINED ON'})")
    if binding["family"] not in seen:
        print("  -> the hardest family is one recovery never trained on. The K2 claim\n"
              "     ('blur binds') was derived from the 3 trained families and does\n"
              "     NOT survive the full registry. Revisit the Stage 2 arm list.")
    elif binding["family"] != "gaussian_blur":
        print(f"  -> K2 needs rewriting: {binding['family']}, not gaussian_blur, is\n"
              "     the binding constraint. Pick Stage 2 arms against it instead.")
    else:
        print("  -> K2 survives the full registry: blur is still the binding\n"
              "     constraint. Stage 2 arm list stands.")
