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
        published={"action": "store_true",
                   "help": "build each arm at its paper's configuration "
                           "(Stage 2a) instead of sweeping widths"},
        seeds={"type": int, "nargs": "+", "default": [0, 1, 2]},
        lam={"type": float, "default": config.AE_LAMBDA_MAX},
        pair_with={"default": None,
                   "help": "reference arm for the paired-by-seed comparison"},
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
    # Stage 2a: width=None makes build_recovery use each class's PUBLISHED dict.
    # These are reference rows -- the ceiling each concept reaches when it is not
    # compressed -- not entries in the compute-bounded comparison.
    widths = [None] if args.published else args.widths

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
          f"{len(widths)} config(s) x {len(args.seeds)} seeds = "
          f"{len(fixed) + len(learned) * len(widths) * len(args.seeds)} models")
    if args.published:
        for a in learned:
            m = models.build_recovery(a, width=None)
            print(f"    {a:8s} {models.count_params(m):>11,} params  "
                  f"{getattr(type(m), 'PUBLISHED', 'default')}")

    rows, raw = [], {"baseline1": b1_raw}

    def score(key, arch, width, seed, module):
        res = engine.eval_cached(module, clf, cache)
        raw[key] = res
        bal = balanced(res)
        n_collapsed = sum(engine.collapsed(v) for v in res.values())
        worst_c, worst_v = engine.worst_case(bal, conds)
        row = {
            "arm": arch, "width": "published" if width is None else width,
            "seed": seed, "key": key,
            "params": models.count_params(module),
            "worst_abs": worst_v, "worst_condition": worst_c,
            "worst_gain": worst_v - b1[worst_c],
            "mCE": engine.corruption_error(bal, b1, conds),
            "mean_gain": float(np.mean([bal[c] - b1[c] for c in conds])),
            "clean_delta": bal[data.CLEAN] - b1[data.CLEAN],
            "trained": n_collapsed < len(all_conds),
            # False => the kept epoch was inside the lambda warmup, so this row
            # is a pure-MSE reconstruction that the classifier never shaped.
            "ce_active": getattr(module, "_ce_active", True),
            "selected_epoch": getattr(module, "_selected_epoch", None),
        }
        row.update(category_gains(bal, b1, conds))
        # Per-category floor. The global floor saturates at chance for every
        # arm; within a category it still separates them, and it is what
        # "robust under any condition" actually means per corruption type.
        by_cat = {}
        for c in conds:
            by_cat.setdefault(config.category_of(data.parse_condition(c)[0]), []).append(c)
        for cat, cc in by_cat.items():
            row[f"worst_{cat}"] = engine.worst_case(bal, cc)[1]
        rows.append(row)
        cats = "  ".join(f"{k.replace('gain_',''):<11s}{v:+.3f}"
                         for k, v in row.items() if k.startswith("gain_"))
        print(f"  -> worst={row['worst_abs']:.3f} on {row['worst_condition']}  "
              f"mCE={row['mCE']:.3f}  mean={row['mean_gain']:+.3f}")
        print(f"     {cats}", flush=True)

    for arch in fixed:
        print(f"\n-- {arch} (non-learned) --", flush=True)
        score(arch, arch, 0, -1, models.build_recovery(arch))

    for arch in learned:
        for w in widths:
            for seed in args.seeds:
                tag = f"lam{args.lam:g}_{args.output}"
                key = f"{arch}_{'published' if w is None else f'w{w}'}_s{seed}"
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

    # Worst-case over the FULL registry saturates: there is always some
    # condition at chance, so every arm reports 0.500 and the metric stops
    # discriminating. Recompute it excluding conditions no arm clears -- only
    # knowable now that every arm has run, which is why this is a second pass
    # rather than part of score().
    unfixable = set(engine.unfixable_conditions(
        [balanced(v) for k, v in raw.items() if k != "baseline1"], conds))
    fixable = [c for c in conds if c not in unfixable]
    print(f"\nunfixable by every arm: {len(unfixable)}/{len(conds)} conditions")
    if len(fixable) < len(conds) // 2:
        print("  !! more than half the grid is unfixable. That usually means the\n"
              "     arms are undertrained, not that the conditions are hopeless.")
    for i, r in enumerate(rows):
        bal = balanced(raw[r["key"]])
        wc, wv = engine.worst_case(bal, fixable)
        r["worst_abs_all"], r["worst_condition_all"] = r["worst_abs"], r["worst_condition"]
        r["worst_abs"], r["worst_condition"] = wv, wc
        r["worst_gain"] = wv - b1[wc] if wc else float("nan")
    df = pd.DataFrame(rows)

    store.save("module_comparison", raw)
    store.save_table("module_comparison", df)

    gain_cols = sorted(c for c in df.columns if c.startswith("gain_"))
    agg = df.groupby(["arm", "width"]).agg(
        n=("key", "size"), params=("params", "first"),
        p_trained=("trained", "mean"),
        ce_used=("ce_active", "mean"),
        worst=("worst_abs", "mean"), worst_sd=("worst_abs", "std"),
        mCE=("mCE", "mean"), mean_gain=("mean_gain", "mean"),
        clean_delta=("clean_delta", "mean"),
        **{c: (c, "mean") for c in gain_cols},
        **{c: (c, "mean") for c in df.columns if c.startswith("worst_")
           and c not in ("worst_abs", "worst_condition", "worst_abs_all",
                         "worst_condition_all", "worst_gain")},
    ).reset_index()
    store.save_table("module_comparison_summary", agg)

    print("\n=== per arm x width ===")
    print(agg.to_string(index=False))

    # --- paired comparison -------------------------------------------------
    # Seed effects are shared across arms: seed 1 was the worst seed for every
    # single arm in Stage 2a. An unpaired test throws that away and needs ~51
    # seeds to resolve a 0.05 mCE gap; pairing on seed needs ~2, because the
    # per-seed difference has sd 0.025 against 0.081 for the raw scores.
    ref = args.pair_with
    if ref is None:
        trained = df[df["arm"].isin(models.LEARNED_NAMES) & df["ce_active"]]
        ref = trained.groupby("arm")["mCE"].mean().idxmin() if len(trained) else None
    if ref is not None and (df["arm"] == ref).any():
        base = df[df["arm"] == ref].set_index("seed")["mCE"]
        print(f"\n=== paired by seed, vs {ref} (negative = better than {ref}) ===")
        print(f"  {'arm':10s}{'d(mCE)':>9s}{'+/-95%':>9s}{'p':>8s}   per-seed")
        for arm, g in df[df["arm"] != ref].groupby("arm"):
            g = g[g["seed"].isin(base.index)]
            if len(g) < 2:
                continue
            d = np.array([r.mCE - base[r.seed] for r in g.itertuples()])
            n = len(d)
            from scipy import stats as _st
            ci = _st.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
            _, pv = _st.ttest_1samp(d, 0.0)
            star = "  *" if pv < .05 else ""
            print(f"  {arm:10s}{d.mean():>+9.3f}{ci:>9.3f}{pv:>8.3f}   "
                  f"{np.round(d, 3).tolist()}{star}")

    print("\n=== read this ===")
    # If nothing trained, every learned arm collapsed to the same constant
    # output and the whole table is one number repeated. Say so before printing
    # a ranking of it -- identical mCE across architectures reads as a finding.
    live = agg[agg["arm"].isin(models.LEARNED_NAMES)]
    if len(live) and live["p_trained"].max() == 0:
        print("  !! NO learned arm trained. Every one collapsed, so the rows below\n"
              "     are the same constant model under different names and the\n"
              "     comparison is void.\n"
              "     Usual cause: --limit / --epochs too small for the lambda ramp.\n"
              "     The CE term reaches full strength before MSE has taught the\n"
              "     module anything, and best-epoch selection then falls back to\n"
              "     the pure-MSE epoch. Rerun on full data at the default epochs.\n")
    # Ranking key: worst-case ONLY while it still discriminates.
    #
    # Over the full registry it usually does not. Every arm bottoms out within a
    # fraction of a point of chance -- the w=4..32 sweep spanned 0.4977 to
    # 0.5000 across all 18 rows -- so sorting on it ranks noise, and it put the
    # best arm in the table (convae w=32, mCE 0.795) sixteenth, below arms at
    # mCE 1.37. mCE is the metric carrying the signal there, so switch keys
    # rather than quietly reporting a meaningless order.
    spread = agg["worst"].max() - agg["worst"].min()
    saturated = spread < 0.01
    if saturated:
        print(f"  [worst-case spans only {spread:.4f} across arms -- it has\n"
              f"   saturated at chance and cannot rank. Ranking by mCE.]\n")
        ranked = agg.sort_values("mCE")
    else:
        ranked = agg.sort_values(["worst", "mCE"], ascending=[False, True])
    for r in ranked.itertuples():
        neg = [c.replace("gain_", "") for c in gain_cols
               if getattr(r, c, 0) is not None and getattr(r, c, 0) < 0]
        flag = f"  HARMS: {', '.join(neg)}" if neg else "  no category harmed"
        ce = getattr(r, "ce_used", 1.0)
        if ce < 1.0:
            flag = (f"  CE LIVE IN ONLY {ce:.0%} OF SEEDS -- the rest kept a "
                    f"pure-MSE epoch") + flag
        print(f"  {r.arm:8s} {str(r.width):<9s} ({r.params:>10,} params)  "
              f"worst={r.worst:.3f}  mCE={r.mCE:.3f}  trains {r.p_trained:.0%}{flag}")

    # identity is excluded from every recommendation: it scores exactly 0.000 on
    # each category by construction, so "no category harmed" is trivially true
    # for it and it would always be crowned. mCE < 1 is the real bar -- that is
    # the point at which an arm beats doing nothing.
    real = ranked[ranked["arm"] != "identity"]
    useful = real[real["mCE"] < 1.0]
    clean = real[[all(getattr(r, c, 0) >= 0 for c in gain_cols)
                  for r in real.itertuples()]]

    if not real.empty:
        b = real.iloc[0]
        print(f"\n  best worst-case (excl. identity): {b['arm']} {b['width']} "
              f"-> {b['worst']:.3f}, mCE={b['mCE']:.3f}")
    print(f"  arms that beat doing nothing (mCE < 1): "
          f"{', '.join(useful['arm'].unique()) or 'NONE'}")

    wcats = sorted(c for c in agg.columns if c.startswith("worst_")
                   and c not in ("worst_sd",))
    if wcats:
        print("\n  per-category floor (global floor saturates; this one does not):")
        head = "  " + f"{'arm':10s}{'width':>6s}" + "".join(
            f"{c.replace('worst_',''):>13s}" for c in wcats)
        print(head)
        for r in ranked.itertuples():
            vals = "".join(f"{getattr(r, c, float('nan')):>13.3f}" for c in wcats)
            print(f"  {r.arm:10s}{str(r.width):>10s}{vals}")

    if clean.empty:
        print("\n  NO arm is non-negative on every category. If the arms trained\n"
              "  properly, that is the gap K3 exists to fill -- check p_trained\n"
              "  and the lam ramp before believing it.")
    else:
        c = clean.iloc[0]
        print(f"\n  best with no harmed category: {c['arm']} {c['width']}.\n"
              "  K3 has to beat this, or this becomes the finding instead.")
