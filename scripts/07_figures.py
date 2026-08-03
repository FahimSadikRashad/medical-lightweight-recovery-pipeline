"""Regenerate every paper figure from saved results.

Reads only results/ and checkpoints/ -- no training. Figures whose experiment
hasn't been run yet are skipped rather than failing, so this is safe to run at
any point while the remaining experiments are still in progress.

    python scripts/07_figures.py
"""
import numpy as np
import torch

from _common import (balanced, config, data, engine, models, parse_args, setup,
                     store, frozen_baseline1)

from robustmed import figures

if __name__ == "__main__":
    args = parse_args()
    train, val, test, info, n_classes = setup(args)
    conditions = data.eval_conditions()

    # --- Fig 1: the corruptions themselves --------------------------------
    figures.fig_corruption_grid(train[0][0])

    b1_raw = store.load_optional(store.BASELINE1)
    if not b1_raw:
        raise SystemExit("no results yet -- start with scripts/01_baseline1_frozen.py")
    b1 = balanced(b1_raw)

    sweep = store.load_optional(store.RECOVERY_SWEEP)
    b2_raw = store.load_optional(store.BASELINE2)
    b2 = balanced(b2_raw) if b2_raw else None

    # --- Fig 2: collapse and recovery -------------------------------------
    hw = str(config.HEADLINE_WIDTH)
    if sweep and hw in sweep:
        rec = balanced(sweep[hw])
        params = models.count_params(models.ConvAE(config.HEADLINE_WIDTH))
        figures.fig_collapse_recovery(
            b1, b2, rec, conditions,
            recovery_label=f"+ Recovery AE (w={config.HEADLINE_WIDTH}, {params:,} params)",
        )

    # --- Fig 3: reconstructions -------------------------------------------
    if config.ae_ckpt(config.HEADLINE_WIDTH).exists():
        ae = engine.load_recovery(config.HEADLINE_WIDTH)
        pairs = data.loader(
            data.Pairs(val, config.TRAIN_CORRUPTIONS, config.TRAIN_SEVERITIES),
            shuffle=True, batch_size=6)
        cor, clean, _ = next(iter(pairs))
        with torch.no_grad():
            recovered = ae(cor.to(engine.DEVICE))
        figures.fig_reconstructions(cor, recovered, clean, config.HEADLINE_WIDTH)

    # --- Fig 4: Pareto (main RQ) ------------------------------------------
    cost = store.load_optional(store.RECOVERY_COST)
    if sweep and cost:
        widths = [w for w in config.AE_WIDTHS if str(w) in sweep and str(w) in cost]
        corrupted = [c for c in conditions if c != data.CLEAN]
        gains, params, lat, clean_pres = [], [], [], []
        for w in widths:
            bal = balanced(sweep[str(w)])
            gains.append(float(np.mean([bal[c] - b1[c] for c in corrupted])))
            params.append(cost[str(w)]["params"])
            lat.append(cost[str(w)]["latency_ms_bs1"])
            clean_pres.append(bal[data.CLEAN])
        figures.fig_pareto(widths, params, gains, lat, clean_pres)

    # --- Fig 5: generalization --------------------------------------------
    figures.fig_generalization(
        compound=store.load_optional(store.COMPOUND),
        transfer=store.load_optional(store.TRANSFER),
    )

    print(f"\nfigures in {config.FIG_DIR}")
    print("results available:", ", ".join(store.available()))
