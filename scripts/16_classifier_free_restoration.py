"""RQ-5 -- classifier-free recovery: does capacity substitute for guidance?

Every arm elsewhere in this project is trained through
CE(frozen_clf(recon), y) -- the mechanism the README calls load-bearing at
~10^4 params ("MSE alone converges to the identity map and recovers
nothing"). This script asks whether that is a fact about the REGIME or about
the LOSS: trained at a real restoration budget (arms.py's PUBLISHED configs,
10^5-10^7 params) with a loss that never sees the classifier -- the same
paradigm NAFNet/PromptIR/MoCE-IR/MIRAGE actually use -- does the module still
recover downstream accuracy?

Two report tables, matching two different literatures:

  restoration quality   PSNR/SSIM per corruption category, the same shape as
                         the all-in-one restoration tables in the
                         MIRAGE/NAFNet/PromptIR papers -- this project's
                         numbers next to that literature's, on medical data.
  downstream accuracy    balanced accuracy through the frozen classifier(s),
                         same metric as every other arm here, compared
                         directly against module_comparison_summary's
                         CE-guided rows for the SAME architecture at the SAME
                         published size -- that comparison IS the H-M2 test.

Training is "all-in-one": every corruption family in the registry, not the
usual 3-family TRAIN_CORRUPTIONS protocol, because that is what the
restoration literature this compares against actually does. Domain stays
medical throughout -- this is not a natural-image benchmark run.

    python scripts/16_classifier_free_restoration.py
    python scripts/16_classifier_free_restoration.py --arms dncnn nafnet span safmn moceir
"""
import numpy as np
import pandas as pd
import torch

from _common import (balanced, config, corruptions, data, engine, models,
                     parse_args, setup, store, frozen_baseline1)


def category_quality(ae, cache, conds, batch_size=256):
    """Mean PSNR/SSIM per corruption category on the cached test grid -- the
    restoration-side counterpart to 13_module_comparison.py's category_gains,
    which does the same grouping for downstream accuracy."""
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    clean_x, _ = cache[data.CLEAN]
    by_cat = {}
    for cond in conds:
        name, _ = data.parse_condition(cond)
        cat = config.category_of(name)
        x_uint8, y = cache[cond]
        ps, ss = [], []
        for i in range(0, len(y), batch_size):
            cor = engine.chunk01(x_uint8[i:i + batch_size])
            cln = engine.chunk01(clean_x[i:i + batch_size])
            with torch.no_grad():
                out = ae(cor) if ae is not None else cor
            ps.append(psnr(out, cln).item())
            ss.append(ssim(out, cln).item())
        by_cat.setdefault(cat, []).append((float(np.mean(ps)), float(np.mean(ss))))
    return {cat: (float(np.mean([v[0] for v in vs])),
                  float(np.mean([v[1] for v in vs])))
            for cat, vs in by_cat.items()}


if __name__ == "__main__":
    args = parse_args(
        arms={"nargs": "+", "default": ["dncnn", "nafnet", "span", "safmn"]},
        ssim_weight={"type": float, "default": 0.5},
        seed={"type": int, "default": 0},
        batch_size={"type": int, "default": None,
                    "help": "override config.BATCH_SIZE for training only. "
                            "moceir measured 7.47GB peak RSS at batch=8, 224px "
                            "(robustmed/moceir.py's verified numbers) -- the "
                            "project default of 128 will exhaust memory. "
                            "Use 8 whenever 'moceir' is in --arms."},
        eval_limit={"type": int, "default": None,
                    "help": "cap TEST/VAL images for the corruption cache -- "
                            "see 13_module_comparison.py's flag of the same name"},
    )
    train, val, test, info, n_classes = setup(args)
    if args.eval_limit:
        if len(test) > args.eval_limit:
            test = data.subset(test, args.eval_limit)
        if len(val) > args.eval_limit:
            val = data.subset(val, args.eval_limit)
    if args.epochs:
        config.AE_EPOCHS = args.epochs

    clf = frozen_baseline1(n_classes)
    all_families = corruptions.names()
    conds = [data.condition(n, s) for n in all_families
             for s in range(config.N_SEVERITIES)]
    all_conds = [data.CLEAN] + conds

    print(f"\ntraining regime: ALL-IN-ONE -- {len(all_families)} families x "
          f"{config.N_SEVERITIES} severities (vs this project's usual "
          f"{len(config.TRAIN_CORRUPTIONS)}-family protocol elsewhere)")

    print(f"\ncaching {len(all_conds)} conditions (test) ...")
    test_cache = engine.make_condition_cache(test, all_conds, limit=args.limit)
    print(f"caching {len(all_conds)} conditions (val, for classifier-free "
          f"model selection) ...")
    val_cache = engine.make_condition_cache(val, all_conds, limit=args.limit)

    b1 = balanced(engine.eval_cached(None, clf, test_cache))
    print(f"  baseline clean={b1[data.CLEAN]:.4f}  "
          f"mean corrupted={np.mean([b1[c] for c in conds]):.4f}")

    # module_comparison_summary is the CE-guided reference table (13_module_
    # comparison.py) -- a CSV, not a JSON result, so store.load_optional
    # doesn't apply here.
    ce_path = config.RESULT_DIR / "module_comparison_summary.csv"
    ce_summary = pd.read_csv(ce_path) if ce_path.exists() else None
    if ce_summary is None:
        print("\nnote: module_comparison_summary.csv not found -- the H-M2 "
              "comparison (classifier-free vs CE-guided, same arch/size) will "
              "be skipped. Run 13_module_comparison.py --published first for it.")

    if "moceir" in args.arms and not args.batch_size:
        print("\n!! 'moceir' is in --arms with no --batch-size override -- it "
              "will train at config.BATCH_SIZE "
              f"({config.BATCH_SIZE}) and is very likely to exhaust memory "
              "(see robustmed/moceir.py: 7.47GB measured at batch=8, 224px; "
              "batch=16 already fails). Pass --batch-size 8, or expect this "
              "to fail.")

    pair_loader = data.loader(
        data.Pairs(train, all_families, list(range(config.N_SEVERITIES))),
        shuffle=True, batch_size=args.batch_size)

    rows = []
    for arch in args.arms:
        tag = f"classfree_ssim{args.ssim_weight:g}"
        key = f"{arch}_published_classfree"
        print(f"\n-- {key} --", flush=True)
        ckpt = config.ae_ckpt(None, seed=args.seed, tag=tag, arch=arch)
        if ckpt.exists():
            ae = engine.load_recovery(None, seed=args.seed, tag=tag, arch=arch)
            print("  resumed from checkpoint")
        else:
            engine.set_seed(args.seed)
            ae = engine.train_restoration(
                None, pair_loader, val_cache, epochs=config.AE_EPOCHS,
                ssim_weight=args.ssim_weight, seed=args.seed, tag=tag, arch=arch)

        params = models.count_params(ae)
        quality = category_quality(ae, test_cache, conds)
        res = engine.eval_cached(ae, clf, test_cache)
        bal = balanced(res)
        mCE = engine.corruption_error(bal, b1, conds)
        mean_gain = float(np.mean([bal[c] - b1[c] for c in conds]))
        avg_psnr = float(np.mean([v[0] for v in quality.values()]))
        avg_ssim = float(np.mean([v[1] for v in quality.values()]))

        row = {"arm": arch, "params": params, "avg_psnr": avg_psnr,
               "avg_ssim": avg_ssim, "mCE": mCE, "mean_gain": mean_gain,
               "clean_delta": bal[data.CLEAN] - b1[data.CLEAN]}
        for cat, (p, s) in quality.items():
            row[f"psnr_{cat}"], row[f"ssim_{cat}"] = p, s
        print(f"  -> params={params:,}  avg_psnr={avg_psnr:.2f}dB  "
              f"avg_ssim={avg_ssim:.4f}  mCE={mCE:.3f}  mean_gain={mean_gain:+.3f}")

        if ce_summary is not None:
            match = ce_summary[(ce_summary["arm"] == arch)
                               & (ce_summary["width"].astype(str) == "published")]
            if len(match):
                ce_mCE = float(match.iloc[0]["mCE"])
                row["ce_guided_mCE"] = ce_mCE
                verdict = "capacity substitutes for guidance (H-M2 supported)" \
                    if mCE <= ce_mCE else "guidance still needed even at scale (H-M2 refuted)"
                print(f"     vs CE-guided (module_comparison, same arch/size): "
                      f"mCE {ce_mCE:.3f} -> classifier-free {mCE:.3f}  [{verdict}]")
        rows.append(row)

    df = pd.DataFrame(rows)
    store.save(store.CLASSIFIER_FREE, rows)
    store.save_table("classifier_free_restoration", df)

    # --- restoration table, in the shared-image's own shape --------------
    cat_cols = sorted({c[5:] for c in df.columns if c.startswith("psnr_")})
    print("\n=== restoration quality (PSNR dB / SSIM), classifier-free training ===")
    header = "  " + f"{'Method':10s}{'Params':>10s}" + "".join(
        f"{c.title():>18s}" for c in cat_cols) + f"{'Average':>18s}"
    print(header)
    for r in df.itertuples():
        cells = "".join(f"{getattr(r, f'psnr_{c}'):>7.2f}/{getattr(r, f'ssim_{c}'):.3f}  "
                        for c in cat_cols)
        print(f"  {r.arm:10s}{r.params:>10,}  {cells}"
              f"{r.avg_psnr:>7.2f}/{r.avg_ssim:.3f}")

    print("\n=== downstream accuracy (through the frozen classifier) ===")
    print(df[["arm", "params", "mCE", "mean_gain", "clean_delta"]
            + (["ce_guided_mCE"] if "ce_guided_mCE" in df.columns else [])
           ].to_string(index=False))

    print("\n=== read this ===")
    if "ce_guided_mCE" in df.columns:
        supported = (df["mCE"] <= df["ce_guided_mCE"]).sum()
        print(f"  H-M2: {supported}/{len(df)} arms did AS WELL OR BETTER "
              f"downstream without any classifier guidance during training.")
        if supported == len(df):
            print("  Capacity substitutes for guidance at published scale -- the "
                  "CE term looks load-bearing only in the ~10^4-param regime.")
        elif supported == 0:
            print("  Guidance helps at every scale tested -- H-M2 refuted, not "
                  "just unconfirmed.")
        else:
            print("  Mixed: which architectures need guidance and which don't "
                  "is itself the finding -- check per-arm rows above, not the count.")
    else:
        print("  Run 13_module_comparison.py --published first, then rerun this "
              "script, to get the H-M2 comparison instead of restoration numbers alone.")
