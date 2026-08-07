"""Stage 2 -- the main experiment: which tiny module design actually works?

Every arm is trained and evaluated identically, then compared on three axes:

  worst-case balanced accuracy   does robustness hold under EVERY condition
  per-category gain              and does it hold for every KIND of corruption
  P(trains)                      does it get there reliably across seeds

against cost measured as parameters and MACs, so the output is a Pareto
frontier rather than a single winner at one arbitrary operating point.

What Stage 1 changed about this stage. The original arm list was chosen because
blur was believed to be the binding constraint. It is not -- over the full
13-family registry blur is the BEST recovered category (+0.172 mean gain), while
photometric corruption is the binding one at -0.056, i.e. recovery leaves the
classifier worse off than the untouched image. So the axis under test is not
high-frequency reconstruction but conditional behaviour: can an architecture
change what it does per input, including doing nothing?

That makes the per-category table the headline output of this script, not an
appendix. An arm that wins on worst-case while still going negative on
photometric has not solved the problem -- it has averaged over it.

Evaluation uses the full registry via a cached corrupted test set, rendered once
and shared by every model. Rendering is the bottleneck, not the GPU.

    python scripts/13_module_comparison.py --limit 256              # smoke
    python scripts/13_module_comparison.py --arms convae dncnn span
    python scripts/13_module_comparison.py --widths 8 16 --seeds 0 1 2
"""
import numpy as np
import pandas as pd

from _common import (balanced, config, corruptions, data, engine, models,
                     parse_args, setup, store, frozen_baseline1)


def category_gains(bal, b1, conds):
    """Mean gain per corruption category -- the table Stage 1 made the headline."""
    out = {}
    for cond in conds:
        name, _ = data.parse_condition(cond)
        out.setdefault(config.category_of(name), []).append(bal[cond] - b1[cond])
    return {f"gain_{k}": float(np.mean(v)) for k, v in out.items()}


if __name__ == "__main__":
    args = parse_args(
        arms={"nargs": "+", "default": list(models.ARM_NAMES)},
        widths={"type": int, "nargs": "+", "default": [4, 16, 32]},
        seeds={"type": int, "nargs": "+", "default": [0, 1, 2]},
        lam={"type": float, "default": config.AE_LAMBDA_MAX},
        output={"default": config.RECOVERY_OUTPUT,
                "choices": list(config.RECOVERY_OUTPUTS)},
    )
    train, val, test, info, n_classes = setup(args)
    if args.epochs:
        config.AE_EPOCHS = args.epochs

    clf = frozen_baseline1(n_classes)
    conds = [data.condition(n, s) for n in corruptions.names()
             for s in range(config.N_SEVERITIES)]
    all_conds = [data.CLEAN] + conds

    print(f"\ncaching {len(all_conds)} conditions (rendered once, reused by "
          f"every arm) ...")
    cache = engine.make_condition_cache(test, all_conds, limit=args.limit)

    print("\n== baseline 1 (no recovery) ==")
    b1_raw = engine.eval_cached(None, clf, cache)
    b1 = balanced(b1_raw)
    print(f"  clean={b1[data.CLEAN]:.4f}  "
          f"mean corrupted={np.mean([b1[c] for c in conds]):.4f}")

    val_probe = engine.make_val_probe(val)
    learned = [a for a in args.arms if a in models.LEARNED_NAMES]
    fixed = [a for a in args.arms if a in models.NON_LEARNED]
    print(f"\ngrid: {len(fixed)} non-learned + {len(learned)} learned x "
          f"{len(args.widths)} widths x {len(args.seeds)} seeds = "
          f"{len(fixed) + len(learned) * len(args.widths) * len(args.seeds)} models")

    rows, raw = [], {"baseline1": b1_raw}

    def score(key, arch, width, seed, module):
        res = engine.eval_cached(module, clf, cache)
        raw[key] = res
        bal = balanced(res)
        n_collapsed = sum(engine.collapsed(v) for v in res.values())
        worst_c, worst_v = engine.worst_case(bal, conds)
        row = {
            "arm": arch, "width": width, "seed": seed, "key": key,
            "params": models.count_params(module),
            "worst_abs": worst_v, "worst_condition": worst_c,
            "worst_gain": worst_v - b1[worst_c],
            "mCE": engine.corruption_error(bal, b1, conds),
            "mean_gain": float(np.mean([bal[c] - b1[c] for c in conds])),
            "clean_delta": bal[data.CLEAN] - b1[data.CLEAN],
            "trained": n_collapsed < len(all_conds),
        }
        row.update(category_gains(bal, b1, conds))
        rows.append(row)
        cats = "  ".join(f"{k.replace('gain_',''):<11s}{v:+.3f}"
                         for k, v in row.items() if k.startswith("gain_"))
        print(f"  -> worst={row['worst_abs']:.3f} on {row['worst_condition']}  "
              f"mCE={row['mCE']:.3f}  mean={row['mean_gain']:+.3f}")
        print(f"     {cats}", flush=True)

    for arch in fixed:
        print(f"\n-- {arch} (non-learned) --", flush=True)
        score(arch, arch, 0, None, models.build_recovery(arch))

    for arch in learned:
        for w in args.widths:
            for seed in args.seeds:
                tag = f"lam{args.lam:g}_{args.output}"
                key = f"{arch}_w{w}_s{seed}"
                print(f"\n-- {key} --", flush=True)
                if config.ae_ckpt(w, seed=seed, tag=tag, arch=arch).exists():
                    module = engine.load_recovery(w, seed=seed, tag=tag, arch=arch)
                    print("  resumed from checkpoint")
                else:
                    engine.set_seed(seed)
                    pair_loader = data.loader(
                        data.Pairs(train, config.TRAIN_CORRUPTIONS,
                                   config.TRAIN_SEVERITIES), shuffle=True)
                    module = engine.train_recovery(
                        w, clf, pair_loader, epochs=config.AE_EPOCHS,
                        lambda_max=args.lam, output=args.output, arch=arch,
                        val_probe=val_probe, seed=seed, tag=tag)
                score(key, arch, w, seed, module)

    df = pd.DataFrame(rows)
    store.save("module_comparison", raw)
    store.save_table("module_comparison", df)

    gain_cols = sorted(c for c in df.columns if c.startswith("gain_"))
    agg = df.groupby(["arm", "width"]).agg(
        n=("seed", "count"), params=("params", "first"),
        p_trained=("trained", "mean"),
        worst=("worst_abs", "mean"), worst_sd=("worst_abs", "std"),
        mCE=("mCE", "mean"), mean_gain=("mean_gain", "mean"),
        clean_delta=("clean_delta", "mean"),
        **{c: (c, "mean") for c in gain_cols},
    ).reset_index()
    store.save_table("module_comparison_summary", agg)

    print("\n=== per arm x width ===")
    print(agg.to_string(index=False))

    print("\n=== read this ===")
    # An arm that wins on the average while still damaging a category has not
    # solved the problem, so rank on worst-case and flag negatives separately.
    ranked = agg.sort_values("worst", ascending=False)
    for r in ranked.itertuples():
        neg = [c.replace("gain_", "") for c in gain_cols
               if getattr(r, c, 0) is not None and getattr(r, c, 0) < 0]
        flag = f"  HARMS: {', '.join(neg)}" if neg else "  no category harmed"
        print(f"  {r.arm:8s} w={r.width:<3d} ({r.params:>7,} params)  "
              f"worst={r.worst:.3f}  mCE={r.mCE:.3f}  trains {r.p_trained:.0%}{flag}")

    best = ranked.iloc[0]
    clean_arms = ranked[[all(getattr(r, c, 0) >= 0 for c in gain_cols)
                         for r in ranked.itertuples()]]
    print(f"\n  best worst-case : {best['arm']} w={int(best['width'])} "
          f"({best['worst']:.3f})")
    if clean_arms.empty:
        print("  NO arm is non-negative on every category. The photometric gap\n"
              "  Stage 1 found is not solved by any existing architecture --\n"
              "  which is the opening K3 exists to fill.")
    else:
        c = clean_arms.iloc[0]
        print(f"  best with no harmed category: {c['arm']} w={int(c['width'])}. "
              f"Check whether K3 still adds anything over it.")
