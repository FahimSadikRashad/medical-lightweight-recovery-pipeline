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


def variant_tag(lambda_max, residual):
    """Distinct filename per variant, so runs never overwrite each other."""
    tag = f"lam{lambda_max:g}"
    return tag + "_res" if residual else tag


if __name__ == "__main__":
    args = parse_args(
        seeds={"type": int, "nargs": "+", "default": config.AE_SEEDS},
        widths={"type": int, "nargs": "+", "default": config.AE_WIDTHS},
        lambdas={"type": float, "nargs": "+", "default": [config.AE_LAMBDA_MAX]},
        residual={"choices": ["false", "true", "both"], "default": "false"},
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

    residuals = {"false": [False], "true": [True], "both": [False, True]}[args.residual]
    val_probe = engine.make_val_probe(val)

    print(f"\ngrid: {len(args.seeds)} seeds x {len(args.widths)} widths x "
          f"{len(args.lambdas)} lambdas x {len(residuals)} residual = "
          f"{len(args.seeds) * len(args.widths) * len(args.lambdas) * len(residuals)} runs")

    rows, raw = [], {}
    for seed in args.seeds:
        # config.SEED drives data.loader's generator, so setting it here is what
        # makes the DATA ORDER vary across seeds too -- not just the weight init.
        config.SEED = seed
        engine.set_seed(seed)
        pair_loader = data.loader(
            data.Pairs(train, config.TRAIN_CORRUPTIONS, config.TRAIN_SEVERITIES),
            shuffle=True)

        for w in args.widths:
            for lam in args.lambdas:
                for res in residuals:
                    tag = variant_tag(lam, res)
                    key = f"s{seed}_w{w}_{tag}"
                    print(f"\n-- {key} --", flush=True)

                    if config.ae_ckpt(w, seed=seed, tag=tag).exists():
                        ae = engine.load_recovery(w, seed=seed, tag=tag)
                        print("  resumed from checkpoint")
                    else:
                        # re-seed per run so runs are individually reproducible
                        engine.set_seed(seed)
                        ae = engine.train_recovery(
                            w, clf, pair_loader, epochs=config.AE_EPOCHS,
                            lambda_max=lam, residual=res,
                            val_probe=val_probe, seed=seed, tag=tag)

                    res_all = engine.eval_conditions(ae, clf, test, conditions)
                    raw[key] = res_all
                    bal = balanced(res_all)
                    n_collapsed = sum(engine.collapsed(v) for v in res_all.values())

                    rows.append({
                        "seed": seed, "width": w, "lam": lam, "residual": res,
                        "params": models.count_params(ae),
                        "mean_gain": float(np.mean([bal[c] - b1[c] for c in corrupted])),
                        "gain_holdout": float(np.mean([bal[c] - b1[c] for c in held])),
                        "clean_delta": bal[data.CLEAN] - b1[data.CLEAN],
                        "collapsed_conds": n_collapsed,
                        "total_collapse": n_collapsed == len(conditions),
                    })
                    r = rows[-1]
                    print(f"  -> mean_gain={r['mean_gain']:+.3f} "
                          f"holdout={r['gain_holdout']:+.3f} "
                          f"clean_delta={r['clean_delta']:+.3f} "
                          f"collapsed={n_collapsed}/{len(conditions)}", flush=True)

    df = pd.DataFrame(rows)
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

    agg = df.groupby(["width", "lam", "residual"]).agg(
        params=("params", "first"),
        n=("seed", "count"),
        gain_mean=("mean_gain", "mean"),
        gain_std=("mean_gain", lambda s: s.std(ddof=1)),
        gain_min=("mean_gain", "min"),
        gain_max=("mean_gain", "max"),
        ci95=("mean_gain", ci95),
        holdout_mean=("gain_holdout", "mean"),
        clean_delta_mean=("clean_delta", "mean"),
        n_collapsed=("total_collapse", "sum"),
    ).reset_index()
    store.save_table("stability_summary", agg)

    print("\n=== aggregated over seeds ===")
    print(agg.to_string(index=False))

    print("\n=== read this ===")
    for r in agg.itertuples():
        sig = "" if np.isnan(r.ci95) else (
            "  SIGNIFICANT" if abs(r.gain_mean) > r.ci95 else
            "  CI SPANS ZERO -- not distinguishable from no recovery")
        print(f"  w={r.width:<3d} lam={r.lam:<5g} res={str(r.residual):5s} "
              f"({r.params:,} params)  gain={r.gain_mean:+.3f} +/-{r.ci95:.3f}  "
              f"collapsed {r.n_collapsed}/{r.n}{sig}")

    worst = agg.loc[agg["n_collapsed"].idxmax()]
    if worst["n_collapsed"] > 0:
        print(f"\nStill collapsing: {int(worst['n_collapsed'])}/{int(worst['n'])} seeds at "
              f"w={int(worst['width'])} lam={worst['lam']:g} "
              f"residual={worst['residual']}.")
        print("If best-epoch selection did not fix it, the collapse happens from the\n"
              "first epoch the CE term is active -- try a lower lambda or residual=True.")
