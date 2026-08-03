"""Evaluate every recovery width in front of the frozen Baseline 1 classifier.

This produces the main-RQ table: recovery gain per width, including on the
held-out severity, plus the clean-input column that shows what the module
costs when there is nothing to fix.

    python scripts/04_evaluate_recovery.py
"""
import numpy as np
import pandas as pd

from _common import (balanced, config, data, engine, models, parse_args, setup,
                     store, frozen_baseline1)

if __name__ == "__main__":
    args = parse_args()
    train, val, test, info, n_classes = setup(args)

    clf = frozen_baseline1(n_classes)
    b1 = balanced(store.load(store.BASELINE1))
    conditions = data.eval_conditions()

    sweep = {}
    for w in config.AE_WIDTHS:
        print(f"\n== width={w} ==")
        ae = engine.load_recovery(w)
        sweep[str(w)] = engine.eval_conditions(ae, clf, test, conditions)
    store.save(store.RECOVERY_SWEEP, sweep)

    # --- main table --------------------------------------------------------
    corrupted_conds = [c for c in conditions if c != data.CLEAN]
    rows = []
    for w in config.AE_WIDTHS:
        bal = balanced(sweep[str(w)])
        gains = [bal[c] - b1[c] for c in corrupted_conds]
        held = [c for c in corrupted_conds
                if data.parse_condition(c)[1] == config.HOLDOUT_SEVERITY]
        rows.append({
            "width": w,
            "params": models.count_params(models.ConvAE(w)),
            "mean_gain": float(np.mean(gains)),
            "mean_gain_holdout_sev": float(np.mean([bal[c] - b1[c] for c in held])),
            "clean_preserved": bal[data.CLEAN],
            "clean_delta": bal[data.CLEAN] - b1[data.CLEAN],
        })
    df = pd.DataFrame(rows)
    store.save_table("recovery_sweep_summary", df)
    print()
    print(df.to_string(index=False))

    best = df.loc[df["mean_gain"].idxmax()]
    print(f"\nbest mean gain: width={int(best['width'])} "
          f"({int(best['params']):,} params, +{best['mean_gain']:.3f})")
    if int(best["width"]) != config.HEADLINE_WIDTH:
        print(f"note: config.HEADLINE_WIDTH is {config.HEADLINE_WIDTH} -- "
              f"update it if {int(best['width'])} is the model you report")
    if best["clean_delta"] < 0:
        print(f"note: on clean input the module costs {best['clean_delta']:+.3f} "
              f"balanced accuracy -- report this, it motivates corruption gating")

    # --- per-condition detail for the headline width ------------------------
    hw = str(config.HEADLINE_WIDTH)
    if hw in sweep:
        bal = balanced(sweep[hw])
        detail = pd.DataFrame([
            {"condition": c, "baseline1": b1[c], "with_recovery": bal[c],
             "gain": bal[c] - b1[c],
             "unseen_severity": data.parse_condition(c)[1] == config.HOLDOUT_SEVERITY
             if c != data.CLEAN else False}
            for c in conditions
        ])
        store.save_table(f"recovery_w{hw}_per_condition", detail)
        print()
        print(detail.to_string(index=False))
