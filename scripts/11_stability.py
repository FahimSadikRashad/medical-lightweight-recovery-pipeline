"""Is the recovery result real, or is it seed luck?

The three-seed run that motivated this script found that BOTH w=4 and w=8
collapse on 1 seed in 3, that the good/bad split does not track capacity, and
that the 95% CI on mean gain spans zero for both widths. The reported +0.23 was
a lucky seed. This script measures the thing that actually varies -- training
stability -- across the grid (seed x width x lambda x residual).

What each axis tests:
  seed      whether any single number is reproducible at all
  width     capacity, the original research question
  lambda    the CE weight. lambda=0 is also the MSE-only ablation, which should
            learn the identity map and recover nothing.
  residual  predicting input+correction anchors the output near the input, which
            structurally forbids the constant-collapse failure mode

Every run is checkpointed and skipped on re-run, so an interrupted session
resumes. Baseline 1 is held FIXED across seeds on purpose: the variance under
test is the recovery module's, not the classifier's. Say so in the paper.

    python scripts/11_stability.py                        # seeds x widths, lambda=default
    python scripts/11_stability.py --lambdas 0 0.5 1.5    # add the lambda sweep
    python scripts/11_stability.py --residual both        # test the structural fix
    python scripts/11_stability.py --seeds 0 1 2 3 4 --widths 8
"""
import numpy as np
import pandas as pd

from _common import (balanced, config, data, engine, models, parse_args, setup,
                     store, frozen_baseline1)


def variant_tag(lambda_max, output):
    """Distinct filename per variant, so runs never overwrite each other."""
    return f"lam{lambda_max:g}_{output}"


if __name__ == "__main__":
    args = parse_args(
        seeds={"type": int, "nargs": "+", "default": config.AE_SEEDS},
        widths={"type": int, "nargs": "+", "default": config.AE_WIDTHS},
        lambdas={"type": float, "nargs": "+", "default": [config.AE_LAMBDA_MAX]},
        arch={"default": "convae"},
        outputs={"nargs": "+", "default": [config.RECOVERY_OUTPUT],
                 "choices": list(config.RECOVERY_OUTPUTS)},
    )
    train, val, test, info, n_classes = setup(args)
    if args.epochs:
        config.AE_EPOCHS = args.epochs

    clf = frozen_baseline1(n_classes)
    b1 = balanced(store.load(store.BASELINE1))
    conditions = data.eval_conditions()
    corrupted = [c for c in conditions if c != data.CLEAN]
    held = [c for c in corrupted
            if data.parse_condition(c)[1] == config.HOLDOUT_SEVERITY]

    val_probe = engine.make_val_probe(val)
    # Score over EVERY corrupted condition. Do not pre-filter on the baseline:
    # a condition where the baseline is at chance is where recovery matters
    # most, not least. Unfixable conditions are identified after the fact.
    scored = list(corrupted)

    print(f"\ngrid: {len(args.seeds)} seeds x {len(args.widths)} widths x "
          f"{len(args.lambdas)} lambdas x {len(args.outputs)} outputs = "
          f"{len(args.seeds) * len(args.widths) * len(args.lambdas) * len(args.outputs)} runs")

    rows, raw = [], {}
    for seed in args.seeds:
        config.SEED = seed
        for w in args.widths:
            for lam in args.lambdas:
                for out_mode in args.outputs:
                    tag = variant_tag(lam, out_mode)
                    key = f"s{seed}_w{w}_{tag}"
                    print(f"\n-- {key} --", flush=True)

                    if config.ae_ckpt(w, seed=seed, tag=tag, arch=args.arch).exists():
                        ae = engine.load_recovery(w, seed=seed, tag=tag, arch=args.arch)
                        print("  resumed from checkpoint")
                    else:
                        # Re-seed AND rebuild the loader per run. Building it once
                        # per seed leaves its generator state advancing across the
                        # width loop, so w=4 and w=32 saw different data orders
                        # within one "seed" -- which is exactly what the seed is
                        # supposed to control.
                        engine.set_seed(seed)
                        pair_loader = data.loader(
                            data.Pairs(train, config.TRAIN_CORRUPTIONS,
                                       config.TRAIN_SEVERITIES),
                            shuffle=True)
                        ae = engine.train_recovery(
                            w, clf, pair_loader, epochs=config.AE_EPOCHS,
                            lambda_max=lam, output=out_mode, arch=args.arch,
                            val_probe=val_probe, seed=seed, tag=tag)

                    res_all = engine.eval_conditions(ae, clf, test, conditions)
                    raw[key] = res_all
                    bal = balanced(res_all)
                    n_collapsed = sum(engine.collapsed(v) for v in res_all.values())
                    worst_c, worst_v = engine.worst_case(bal, scored)

                    rows.append({
                        "seed": seed, "width": w, "lam": lam, "output": out_mode,
                        "arch": args.arch, "params": models.count_params(ae),
                        # headline: the floor, not the average
                        "worst_abs": worst_v,
                        "worst_gain": worst_v - b1[worst_c] if worst_c else float("nan"),
                        "worst_condition": worst_c,
                        "mCE": engine.corruption_error(bal, b1, scored),
                        # secondary
                        "mean_gain": float(np.mean([bal[c] - b1[c] for c in corrupted])),
                        "gain_holdout": float(np.mean([bal[c] - b1[c] for c in held])),
                        "clean_delta": bal[data.CLEAN] - b1[data.CLEAN],
                        "collapsed_conds": n_collapsed,
                        "total_collapse": n_collapsed == len(conditions),
                        "trained": n_collapsed < len(conditions),
                    })
                    r = rows[-1]
                    print(f"  -> WORST={r['worst_abs']:.3f} on {r['worst_condition']} "
                          f"(gain {r['worst_gain']:+.3f})  mCE={r['mCE']:.3f}  "
                          f"mean_gain={r['mean_gain']:+.3f}  "
                          f"collapsed={n_collapsed}/{len(conditions)}", flush=True)

    df = pd.DataFrame(rows)
    unfixable = engine.unfixable_conditions(
        [balanced(v) for v in raw.values()], corrupted)
    if unfixable:
        print(f"\nunfixable ({len(unfixable)}/{len(corrupted)}): no run got above "
              f"chance here -- reported as its own group, not excluded from the "
              f"worst case\n  {unfixable}")
    store.save(store.STABILITY, raw)
    store.save_table("stability_runs", df)
    print("\n=== per run ===")
    print(df.to_string(index=False))

    # --- aggregate: the numbers that belong in the paper -------------------
    def ci95(s):
        """Half-width of the 95% CI. With n=3 this is wide -- that IS the finding."""
        n = len(s)
        if n < 2:
            return float("nan")
        from scipy import stats
        return float(stats.t.ppf(0.975, n - 1) * s.std(ddof=1) / np.sqrt(n))

    # Aggregate over ALL seeds, and separately over the ones that trained.
    # Pooling them averages a bimodal distribution: a dead run contributes a
    # fixed, arch-independent number, so the pooled mean mostly measures the
    # failure rate and its CI always spans zero. Both columns belong in the
    # paper -- P(trains) is a property of the module, not a nuisance.
    agg = df.groupby(["arch", "width", "lam", "output"]).agg(
        params=("params", "first"),
        n=("seed", "count"),
        p_trained=("trained", "mean"),
        worst_mean=("worst_abs", "mean"),
        worst_ci95=("worst_abs", ci95),
        mCE_mean=("mCE", "mean"),
        gain_mean=("mean_gain", "mean"),
        gain_std=("mean_gain", lambda s: s.std(ddof=1)),
        ci95=("mean_gain", ci95),
        holdout_mean=("gain_holdout", "mean"),
        clean_delta_mean=("clean_delta", "mean"),
        n_collapsed=("total_collapse", "sum"),
    ).reset_index()

    live = df[df["trained"]]
    if len(live):
        agg_live = live.groupby(["arch", "width", "lam", "output"]).agg(
            n_live=("seed", "count"),
            worst_live=("worst_abs", "mean"),
            worst_live_ci95=("worst_abs", ci95),
            gain_live=("mean_gain", "mean"),
            gain_live_ci95=("mean_gain", ci95),
        ).reset_index()
        agg = agg.merge(agg_live, on=["arch", "width", "lam", "output"], how="left")
    store.save_table("stability_summary", agg)

    print("\n=== aggregated over seeds ===")
    print(agg.to_string(index=False))

    print("\n=== read this ===")
    for r in agg.itertuples():
        ci = getattr(r, "worst_live_ci95", float("nan"))
        sig = "" if np.isnan(ci) else (
            "" if getattr(r, "worst_live", 0) - ci > 0.5 else
            "  CI TOUCHES CHANCE")
        print(f"  {r.arch} w={r.width:<3d} lam={r.lam:<5g} out={r.output:8s} "
              f"({r.params:,} params)  trains {r.p_trained:.0%}  "
              f"worst={getattr(r, 'worst_live', float('nan')):.3f}+/-{ci:.3f}  "
              f"mCE={r.mCE_mean:.3f}{sig}")

    dead = agg.loc[agg["n_collapsed"].idxmax()] if len(agg) else None
    if dead is not None and dead["n_collapsed"] > 0:
        print(f"\nStill collapsing: {int(dead['n_collapsed'])}/{int(dead['n'])} seeds at "
              f"{dead['arch']} w={int(dead['width'])} lam={dead['lam']:g} "
              f"output={dead['output']}.")
        print("Check the grad_norm column in the per-epoch history. Exactly 0.0 means\n"
              "the output saturated and no gradient reaches the weights -- rerun with\n"
              "--outputs residual (or sigmoid), not a lower lambda.")
