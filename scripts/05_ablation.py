"""Ablation: is the learned module doing something a trivial filter can't?

Rows compared on identical conditions:
  1. Baseline 1, no recovery              -- the floor
  2. Baseline 2, augmentation             -- Paper A, requires retraining
  3. 3x3 box denoiser, 0 params           -- non-learned control
  4. Recovery AE (headline width)         -- ours
  5. Recovery AE + Baseline 2 classifier  -- are the two mechanisms complementary?

Row 3 is what makes row 4 a claim about learning. Row 5 matters because if
recovery only matches augmentation, the contribution is weak.

    python scripts/05_ablation.py
"""
import pandas as pd

from _common import (balanced, config, data, engine, models, parse_args, setup,
                     store, frozen_baseline1)

if __name__ == "__main__":
    args = parse_args()
    train, val, test, info, n_classes = setup(args)

    clf = frozen_baseline1(n_classes)
    ae = engine.load_recovery(config.HEADLINE_WIDTH)
    conditions = data.eval_conditions()

    table = {"condition": conditions}
    table["b1_no_recovery"] = [balanced(store.load(store.BASELINE1))[c] for c in conditions]

    b2_raw = store.load_optional(store.BASELINE2)
    if b2_raw:
        table["b2_augmented"] = [balanced(b2_raw)[c] for c in conditions]

    print("\n== non-learned control: 3x3 box denoiser ==")
    box = models.BoxDenoiser().to(engine.DEVICE)
    box_res = engine.eval_conditions(box, clf, test, conditions)
    table["box_denoiser"] = [box_res[c]["balanced_accuracy"] for c in conditions]

    print(f"\n== recovery AE (w={config.HEADLINE_WIDTH}) ==")
    ae_res = engine.eval_conditions(ae, clf, test, conditions)
    table["recovery_ae"] = [ae_res[c]["balanced_accuracy"] for c in conditions]

    df = pd.DataFrame(table)
    store.save(store.ABLATION, {"conditions": conditions,
                                "box_denoiser": box_res, "recovery_ae": ae_res})
    store.save_table("ablation", df)
    print()
    print(df.to_string(index=False))

    print(f"\nlearned vs non-learned, mean over corrupted conditions: "
          f"{df['recovery_ae'][1:].mean() - df['box_denoiser'][1:].mean():+.3f}")

    # --- is recovery complementary to augmentation? ------------------------
    if config.BASELINE2_CKPT.exists():
        print("\n== recovery AE in front of the AUGMENTED classifier ==")
        clf2 = models.Classifier(n_classes).to(engine.DEVICE)
        engine.load_ckpt(config.BASELINE2_CKPT, clf2)
        clf2.freeze()

        # Sanity check first: with no AE, clf2 must reproduce stage 02's numbers.
        # If it doesn't, something in the eval path has drifted -- stop and look.
        sanity = engine.evaluate(clf2, data.condition_loader(test, data.CLEAN))
        expected = balanced(b2_raw)[data.CLEAN] if b2_raw else None
        print(f"sanity: clf2 clean bal={sanity['balanced_accuracy']:.4f}"
              + (f" (stage 02 recorded {expected:.4f})" if expected else ""))

        stacked = engine.eval_conditions(ae, clf2, test, conditions)
        store.save(store.STACKED, stacked)
        sdf = pd.DataFrame([
            {"condition": c,
             "b2_aug_only": balanced(b2_raw)[c],
             "ae_plus_aug": stacked[c]["balanced_accuracy"],
             "complementary_gain": stacked[c]["balanced_accuracy"] - balanced(b2_raw)[c]}
            for c in conditions
        ]) if b2_raw else None
        if sdf is not None:
            store.save_table("recovery_plus_aug", sdf)
            print()
            print(sdf.to_string(index=False))
            print("\nReminder: the AE was trained against Baseline 1, so a drop here is"
                  "\nthe classifier-coupling limitation, not a bug.")
    else:
        print("\nskipping the stacked row -- run scripts/02_baseline2_augmented.py first")
