"""Generalization beyond the training corruptions.

Sub-Q1  compound corruptions -- two corruptions stacked, a fault mode the
        module never trained on and that isn't in the MedMNIST-C registry.
Sub-Q2  cross-domain transfer -- Kermany chest X-rays: different dataset,
        different acquisition, same task. Needs KERMANY_TEST_DIR.

Either sub-question can be run alone; each writes its own result file.

    python scripts/06_generalization.py
    python scripts/06_generalization.py --skip-transfer
"""
import pandas as pd

from _common import (config, data, engine, parse_args, setup, store,
                     frozen_baseline1)

if __name__ == "__main__":
    args = parse_args(
        skip_compound={"action": "store_true"},
        skip_transfer={"action": "store_true"},
    )
    train, val, test, info, n_classes = setup(args)

    clf = frozen_baseline1(n_classes)
    ae = engine.load_recovery(config.HEADLINE_WIDTH)

    # --- Sub-Q1: compound corruptions -------------------------------------
    if not args.skip_compound:
        print("\n== Sub-Q1: unseen compound corruptions ==")
        no_rec = engine.eval_compound(None, clf, test)
        with_rec = engine.eval_compound(ae, clf, test)
        compound = {
            k: {"no_recovery": no_rec[k]["balanced_accuracy"],
                "with_recovery": with_rec[k]["balanced_accuracy"]}
            for k in no_rec
        }
        for k, v in compound.items():
            print(f"  {k:38s} {v['no_recovery']:.3f} -> {v['with_recovery']:.3f}  "
                  f"({v['with_recovery'] - v['no_recovery']:+.3f})")
        store.save(store.COMPOUND, compound)
        df = pd.DataFrame([{"compound": k, **v,
                            "gain": v["with_recovery"] - v["no_recovery"]}
                           for k, v in compound.items()])
        store.save_table("compound_corruptions", df)

    # --- Sub-Q2: Kermany transfer -----------------------------------------
    if args.skip_transfer:
        raise SystemExit(0)

    if not config.KERMANY_TEST_DIR:
        print("\nKERMANY_TEST_DIR not set -- skipping transfer."
              "\nRun: python scripts/download_data.py --kermany")
        raise SystemExit(0)

    print("\n== Sub-Q2: Kermany cross-domain transfer ==")
    items = data.kermany_items()
    if args.limit:
        items = items[::max(1, len(items) // args.limit)]

    def run(recovery, corruption, severity):
        ds = data.Kermany(items, corruption, severity)
        return engine.evaluate(engine.Pipeline(recovery, clf),
                               data.loader(ds))["balanced_accuracy"]

    transfer = {
        "clean": {"no_recovery": run(None, None, None),
                  "with_recovery": run(ae, None, None)},
        f"{config.TRANSFER_CORRUPTION}_sev{config.TRANSFER_SEVERITY}": {
            "no_recovery": run(None, config.TRANSFER_CORRUPTION, config.TRANSFER_SEVERITY),
            "with_recovery": run(ae, config.TRANSFER_CORRUPTION, config.TRANSFER_SEVERITY)},
    }
    for k, v in transfer.items():
        print(f"  {k:26s} {v['no_recovery']:.3f} -> {v['with_recovery']:.3f}  "
              f"({v['with_recovery'] - v['no_recovery']:+.3f})")
    store.save(store.TRANSFER, transfer)
    df = pd.DataFrame([{"condition": k, **v, "gain": v["with_recovery"] - v["no_recovery"]}
                       for k, v in transfer.items()])
    store.save_table("kermany_transfer", df)

    print("\nCompare the corrupted-condition gain here against the in-domain gain in"
          "\nrecovery_wN_per_condition.csv -- similar gains are the transfer claim.")
