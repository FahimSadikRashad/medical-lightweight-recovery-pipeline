"""Stage 2b -- which mechanism buys robustness, at a fixed COMPUTE budget?

Stage 2a compared whole architectures spanning 14k to 29M parameters. Two of the
five never completed a CE-active run, so their mechanisms went unmeasured, and
every difference that did appear was confounded with depth, width and training
stability. This stage removes all of that: one skeleton, one budget, one
mechanism switched on at a time, with each variant's width tuned to the same
budget.

The budget is COMPUTE, not parameters, and at 224px that distinction is not
pedantic. Measured on one CPU thread: convae at 14,067 params costs 55 MMACs and
1.2 ms, while a same-resolution mechanism variant at 13,773 params costs 674
MMACs and 38 ms -- 12x the compute and 33x the latency for the same parameter
count, because convae downsamples twice and the others do not. Matching
parameters would hand the same-resolution mechanisms a 12x compute advantage and
call it a fair test. --budget-kind params is available for the capacity
question, but it is not the efficiency question.

It is analysed PAIRED BY SEED against the plain skeleton. Seed effects are
shared across variants -- in Stage 2a seed 1 was the worst seed for every single
arm -- so an unpaired test discards most of the power. Measured there: per-seed
differences have sd 0.025 against 0.081 for the raw scores, which is 51 seeds
versus 2 to resolve a 0.05 mCE gap.

The contrast to read first is channel_attention vs safm. Photometric corruption
is the binding category and is a global intensity remap; channel_attention is
the only purely global mechanism here, safm is global-ish plus spatially
selective. Stage 2a suggested safm wins, and this is the controlled test.

    python scripts/14_mechanism_ablation.py --seeds 0 1 2
    python scripts/14_mechanism_ablation.py --budget 4000 --seeds 0 1 2 3 4 5
"""
import numpy as np
import pandas as pd
from scipy import stats

from _common import (balanced, config, corruptions, data, engine, models,
                     parse_args, setup, store, frozen_baseline1)

from robustmed import mechanisms as M


if __name__ == "__main__":
    args = parse_args(
        mechanisms={"nargs": "+", "default": list(M.MECHANISMS)},
        seeds={"type": int, "nargs": "+", "default": [0, 1, 2]},
        budget={"type": int, "default": 14067},   # convae published
        budget_kind={"default": "macs", "choices": ["macs", "params"],
                     "help": "match compute (default) or parameters. At 224px "
                             "equal parameters is NOT equal cost: convae is 12x "
                             "cheaper in MACs than a same-resolution variant of "
                             "the same size."},
        blocks={"type": int, "default": 2},
        lam={"type": float, "default": config.AE_LAMBDA_MAX},
        output={"default": config.RECOVERY_OUTPUT,
                "choices": list(config.RECOVERY_OUTPUTS)},
        eval_limit={"type": int, "default": None,
                    "help": "cap TEST images used for the corruption cache -- "
                            "independent of --limit's smoke-run banner. Needed "
                            "whenever a dataset's test split is too large to "
                            "cache at this resolution (e.g. bloodmnist's 3,421 "
                            "images vs pneumoniamnist's 624); try 624 to match."},
    )
    train, val, test, info, n_classes = setup(args)
    if args.eval_limit and len(test) > args.eval_limit:
        # A dataset's TEST split can be far larger than the one this pipeline was
        # tuned against: bloodmnist's ~3,421 images would build a ~29 GB
        # corruption cache at 224px, over the entire Kaggle disk quota by
        # itself. This caps a real evaluation's N deliberately -- distinct from
        # --limit's smoke-run banner -- so the number is reportable, just on a
        # capped sample, and comparable in size to pneumoniamnist's own N=624.
        est_gb = args.eval_limit * 3 * config.IMAGE_SIZE ** 2 * 66 / 1e9
        print(f"eval-limit: capping test split {len(test)} -> {args.eval_limit} "
              f"images (est. cache ~{est_gb:.1f} GB)")
        test = data.subset(test, args.eval_limit)
    if args.epochs:
        config.AE_EPOCHS = args.epochs

    clf = frozen_baseline1(n_classes)
    conds = [data.condition(n, s) for n in corruptions.names()
             for s in range(config.N_SEVERITIES)]
    all_conds = [data.CLEAN] + conds

    print(f"\ncaching {len(all_conds)} conditions ...")
    cache = engine.make_condition_cache(test, all_conds, limit=args.limit)
    b1 = balanced(engine.eval_cached(None, clf, cache))
    print(f"  baseline clean={b1[data.CLEAN]:.4f}  "
          f"mean corrupted={np.mean([b1[c] for c in conds]):.4f}")

    from robustmed import efficiency
    if args.budget_kind == "macs":
        target = args.budget if args.budget != 14067 else efficiency.macs(
            models.build_recovery("convae", width=16, device=None))
        widths = {m: M.fit_macs(m, target, args.blocks) for m in args.mechanisms}
        print(f"\nbudget {target/1e6:.1f} MMACs @ {config.IMAGE_SIZE}px, "
              f"{args.blocks} blocks:")
    else:
        target = args.budget
        widths = {m: M.fit_width(m, target, args.blocks) for m in args.mechanisms}
        print(f"\nbudget {target:,} params, {args.blocks} blocks:")
    for m, w in widths.items():
        net = M.MechanismNet(width=w, mechanism=m, blocks=args.blocks)
        print(f"  {m:20s} width {w:>3d}  {models.count_params(net):>8,} params  "
              f"{(efficiency.macs(net) or 0)/1e6:>7.1f} MMACs")

    val_probe = engine.make_val_probe(val)
    rows = []
    for mech in args.mechanisms:
        for seed in args.seeds:
            tag = f"mech_{mech}_b{args.blocks}_{args.output}"
            print(f"\n-- {mech} seed {seed} --", flush=True)
            ck = config.ae_ckpt(widths[mech], seed=seed, tag=tag, arch="mech")
            if ck.exists():
                net = engine.load_recovery(widths[mech], seed=seed, tag=tag,
                                           arch="mech")
                print("  resumed from checkpoint")
            else:
                engine.set_seed(seed)
                pair_loader = data.loader(
                    data.Pairs(train, config.TRAIN_CORRUPTIONS,
                               config.TRAIN_SEVERITIES), shuffle=True)
                net = engine.train_recovery(
                    widths[mech], clf, pair_loader, epochs=config.AE_EPOCHS,
                    lambda_max=args.lam, output=args.output, arch="mech",
                    val_probe=val_probe, seed=seed, tag=tag,
                    mechanism=mech, blocks=args.blocks)

            res = engine.eval_cached(net, clf, cache)
            bal = balanced(res)
            row = {"mechanism": mech, "seed": seed, "width": widths[mech],
                   "params": models.count_params(net),
                   "mCE": engine.corruption_error(bal, b1, conds),
                   "mean_gain": float(np.mean([bal[c] - b1[c] for c in conds])),
                   "clean_delta": bal[data.CLEAN] - b1[data.CLEAN],
                   "ce_active": getattr(net, "_ce_active", True)}
            for c in conds:
                row.setdefault("_", None)
            by_cat = {}
            for c in conds:
                by_cat.setdefault(config.category_of(data.parse_condition(c)[0]),
                                  []).append(bal[c] - b1[c])
            for cat, v in by_cat.items():
                row[f"gain_{cat}"] = float(np.mean(v))
            row.pop("_", None)
            rows.append(row)
            print(f"  -> mCE={row['mCE']:.3f}  mean={row['mean_gain']:+.3f}  "
                  f"photometric={row.get('gain_photometric', float('nan')):+.3f}"
                  f"{'' if row['ce_active'] else '   [CE NEVER LIVE]'}", flush=True)

    df = pd.DataFrame(rows)
    store.save_table("mechanism_ablation", df)

    gain_cols = sorted(c for c in df.columns if c.startswith("gain_"))
    agg = df.groupby("mechanism").agg(
        n=("seed", "count"), width=("width", "first"), params=("params", "first"),
        ce_used=("ce_active", "mean"), mCE=("mCE", "mean"),
        mean_gain=("mean_gain", "mean"), clean_delta=("clean_delta", "mean"),
        **{c: (c, "mean") for c in gain_cols}).reset_index()
    store.save_table("mechanism_ablation_summary", agg)
    print("\n=== per mechanism ===")
    print(agg.to_string(index=False))

    # --- paired against the plain skeleton ---------------------------------
    print("\n=== paired by seed vs `plain` (negative = mechanism helps) ===")
    if "plain" in df["mechanism"].values:
        base = df[df["mechanism"] == "plain"].set_index("seed")
        print(f"  {'mechanism':20s}{'d(mCE)':>9s}{'+/-95%':>9s}{'p':>8s}"
              f"{'d(photometric)':>16s}")
        for mech, g in df[df["mechanism"] != "plain"].groupby("mechanism"):
            g = g[g["seed"].isin(base.index)]
            if len(g) < 2:
                continue
            d = np.array([r.mCE - base.loc[r.seed, "mCE"] for r in g.itertuples()])
            dp = np.array([getattr(r, "gain_photometric", np.nan)
                           - base.loc[r.seed, "gain_photometric"]
                           for r in g.itertuples()])
            ci = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            _, pv = stats.ttest_1samp(d, 0.0)
            print(f"  {mech:20s}{d.mean():>+9.3f}{ci:>9.3f}{pv:>8.3f}"
                  f"{np.nanmean(dp):>+16.3f}{'  *' if pv < .05 else ''}")

    print("\n=== read this ===")
    best = agg.loc[agg["mCE"].idxmin()]
    print(f"  lowest mCE : {best['mechanism']} ({best['mCE']:.3f})")
    ph = agg.loc[agg["gain_photometric"].idxmax()] if "gain_photometric" in agg else None
    if ph is not None:
        print(f"  best photometric: {ph['mechanism']} "
              f"({ph['gain_photometric']:+.3f})"
              f"{'  -- still negative, K3 needs a dedicated branch' if ph['gain_photometric'] < 0 else ''}")
    dead = agg[agg["ce_used"] < 1.0]
    if len(dead):
        print(f"  CE never live for: {', '.join(dead['mechanism'])} -- those rows\n"
              f"  are pure-MSE and say nothing about the mechanism.")
