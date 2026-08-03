"""Train the recovery autoencoder at every width in the capacity sweep.

Loss = MSE(recon, clean) + lambda * CE(frozen_clf(recon), label), with lambda
ramping from 0 after a warmup. The CE term is what makes this work: MSE alone
converges to the identity map and recovers nothing.

Generalization guard: trained on TRAIN_SEVERITIES (0-2) only, with severity 4
held out so the evaluation includes a severity the module never saw.

    python scripts/03_train_recovery.py
"""
from _common import config, data, engine, models, parse_args, setup, store, frozen_baseline1

if __name__ == "__main__":
    args = parse_args()
    train, val, test, info, n_classes = setup(args)
    if args.epochs:
        config.AE_EPOCHS = args.epochs

    clf = frozen_baseline1(n_classes)
    print("frozen Baseline 1 loaded -- it guides the perceptual loss")

    pair_loader = data.loader(
        data.Pairs(train, config.TRAIN_CORRUPTIONS, config.TRAIN_SEVERITIES),
        shuffle=True,
    )
    print(f"training corruptions: {config.TRAIN_CORRUPTIONS}")
    print(f"training severities:  {config.TRAIN_SEVERITIES} "
          f"(severity {config.HOLDOUT_SEVERITY} held out)")

    print("\n== capacity sweep ==")
    for w in config.AE_WIDTHS:
        print(f"width={w:3d}  params={models.count_params(models.ConvAE(w)):,}")

    for w in config.AE_WIDTHS:
        print(f"\n-- training width={w} --")
        engine.train_recovery(w, clf, pair_loader, epochs=config.AE_EPOCHS)

    print("\n== reconstruction quality (report, but do not select on it) ==")
    val_pairs = data.loader(
        data.Pairs(val, config.TRAIN_CORRUPTIONS, config.TRAIN_SEVERITIES))
    recon = {}
    for w in config.AE_WIDTHS:
        ae = engine.load_recovery(w)
        recon[str(w)] = engine.reconstruction_quality(ae, val_pairs)
        print(f"width={w:3d}  PSNR={recon[str(w)]['psnr']:.2f}  "
              f"SSIM={recon[str(w)]['ssim']:.4f}")
    store.save(store.RECONSTRUCTION, recon)

    print("\n== cost ==")
    cost = {}
    for w in config.AE_WIDTHS:
        ae = engine.load_recovery(w)
        cost[str(w)] = {
            "params": models.count_params(ae),
            "latency_ms_bs1": engine.latency_ms(ae, batch_size=1),
            "latency_ms_bs128": engine.latency_ms(ae, batch_size=128),
        }
        c = cost[str(w)]
        print(f"width={w:3d}  params={c['params']:,}  "
              f"{c['latency_ms_bs1']:.4f} ms/img (bs=1)  "
              f"{c['latency_ms_bs128']:.4f} ms/img (bs=128)")
    store.save(store.RECOVERY_COST, cost)
