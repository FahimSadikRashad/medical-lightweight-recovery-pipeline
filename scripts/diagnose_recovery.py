"""Why is the recovery AE producing no gain? Answer it with output statistics.

Run this when a recovery width shows ~0.000 gain or pins the pipeline at chance.
The failure modes it separates:

  constant collapse   AE ignores its input and emits one fixed image. Diagnosed
                      by std over the batch ~0 and input_sensitivity ~0.
  clamp saturation    ConvAE ends in clamp(0,1), which has ZERO gradient outside
                      the range. If most pre-clamp pixels are out of range the
                      module cannot learn -- diagnosed by clamped_frac.
  dead weights        training never moved the decoder. Diagnosed by weight_std.
  adversarial degenerate
                      output is input-dependent and in range, but the classifier
                      still predicts one class -- the CE term found a shortcut
                      rather than a restoration. Diagnosed by pred_counts with
                      good input_sensitivity.

    python scripts/diagnose_recovery.py
    python scripts/diagnose_recovery.py --width 4

Reads only checkpoints, trains nothing. Safe to run mid-project.
"""
import numpy as np
import torch

from _common import config, data, engine, models, parse_args, setup, frozen_baseline1


@torch.no_grad()
def probe(ae, clf, pair_loader, n_batches=4):
    """Statistics of the AE's output over a few real batches."""
    ae.eval()
    stats = {k: [] for k in ("out_mean", "out_std", "batch_std", "clamped_frac",
                             "sensitivity", "mse_to_clean", "mse_to_input")}
    preds = []

    for i, (cor, clean, y) in enumerate(pair_loader):
        if i >= n_batches:
            break
        cor, clean = cor.to(engine.DEVICE), clean.to(engine.DEVICE)

        # pre-clamp, to see whether clamp() is eating the gradient
        raw = ae.dec(ae.enc(cor))
        if ae.residual:
            raw = cor + raw
        out = torch.clamp(raw, 0.0, 1.0)

        stats["out_mean"].append(out.mean().item())
        stats["out_std"].append(out.std().item())
        # std ACROSS the batch at each pixel: ~0 means every image maps to the
        # same output, which is the constant-collapse signature
        stats["batch_std"].append(out.std(dim=0).mean().item())
        stats["clamped_frac"].append(((raw < 0) | (raw > 1)).float().mean().item())
        stats["mse_to_clean"].append(((out - clean) ** 2).mean().item())
        stats["mse_to_input"].append(((out - cor) ** 2).mean().item())

        # does the output actually depend on the input?
        if cor.size(0) >= 2:
            half = cor.size(0) // 2
            a, b = ae(cor[:half]), ae(cor[half:half * 2])
            stats["sensitivity"].append((a - b).abs().mean().item())

        preds += clf(out).argmax(1).cpu().tolist()

    out = {k: float(np.mean(v)) for k, v in stats.items() if v}
    vals, counts = np.unique(preds, return_counts=True)
    out["pred_counts"] = {int(k): int(v) for k, v in zip(vals, counts)}
    return out


def weight_stats(ae):
    return {
        "weight_std": float(np.mean([p.std().item() for p in ae.parameters()
                                     if p.numel() > 1])),
        "dec_last_std": float(list(ae.dec.parameters())[-2].std().item()),
    }


if __name__ == "__main__":
    args = parse_args(width={"type": int, "default": None})
    train, val, test, info, n_classes = setup(args)

    clf = frozen_baseline1(n_classes)

    # Identity control: what does the classifier do with NO recovery at all?
    # Every AE number below is only meaningful relative to this.
    pairs = data.loader(
        data.Pairs(val, config.TRAIN_CORRUPTIONS, config.TRAIN_SEVERITIES))
    clean_m = engine.evaluate(clf, data.condition_loader(val, data.CLEAN))
    print(f"classifier alone, clean val: bal={clean_m['balanced_accuracy']:.4f} "
          f"preds={clean_m['pred_counts']}")
    if engine.collapsed(clean_m):
        print("  !! the CLASSIFIER itself is collapsed on clean data -- the AE is\n"
              "     not the problem. Retrain Baseline 1 before reading anything below.")

    widths = [args.width] if args.width else config.AE_WIDTHS
    rows = []
    for w in widths:
        if not config.ae_ckpt(w).exists():
            print(f"\nw={w}: no checkpoint, skipping")
            continue
        ae = engine.load_recovery(w)
        s = probe(ae, clf, pairs)
        s.update(weight_stats(ae))
        s["width"] = w
        s["params"] = models.count_params(ae)
        rows.append(s)

        print(f"\n== w={w} ({s['params']:,} params) ==")
        print(f"  output mean/std        {s['out_mean']:.4f} / {s['out_std']:.4f}")
        print(f"  std across batch       {s['batch_std']:.5f}   "
              f"{'<-- CONSTANT COLLAPSE' if s['batch_std'] < 1e-3 else ''}")
        print(f"  input sensitivity      {s['sensitivity']:.5f}   "
              f"{'<-- output ignores input' if s['sensitivity'] < 1e-3 else ''}")
        print(f"  pre-clamp out of range {s['clamped_frac']:.1%}   "
              f"{'<-- clamp is eating the gradient' if s['clamped_frac'] > 0.5 else ''}")
        print(f"  mse to clean           {s['mse_to_clean']:.5f}")
        print(f"  mse to input           {s['mse_to_input']:.5f}   "
              f"{'<-- learned the identity map' if s['mse_to_input'] < 1e-4 else ''}")
        print(f"  weight std             {s['weight_std']:.5f} "
              f"(decoder last {s['dec_last_std']:.5f})")
        print(f"  classifier preds       {s['pred_counts']}   "
              f"{'<-- single class' if len(s['pred_counts']) == 1 else ''}")

    # --- verdict ----------------------------------------------------------
    print("\n" + "=" * 68)
    for s in rows:
        w = s["width"]
        if s["batch_std"] < 1e-3 or s["sensitivity"] < 1e-3:
            v = "CONSTANT COLLAPSE -- AE emits a fixed image regardless of input"
        elif s["clamped_frac"] > 0.5:
            v = "CLAMP SATURATION -- most pixels outside [0,1], gradient is zero"
        elif s["mse_to_input"] < 1e-4:
            v = "IDENTITY MAP -- the CE term never took effect (check lambda ramp)"
        elif len(s["pred_counts"]) == 1:
            v = "ADVERSARIAL DEGENERATE -- output varies but classifier sees one class"
        else:
            v = "output looks healthy -- the problem is elsewhere"
        print(f"  w={w:3d}  {v}")
    print("=" * 68)

    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        w = rows[0]["width"]
        ae = engine.load_recovery(w)
        cor, clean, _ = next(iter(data.loader(
            data.Pairs(val, config.TRAIN_CORRUPTIONS, config.TRAIN_SEVERITIES),
            shuffle=True, batch_size=6)))
        with torch.no_grad():
            rec = ae(cor.to(engine.DEVICE)).cpu()

        fig, axes = plt.subplots(3, 6, figsize=(12, 6))
        for j in range(6):
            for i, (t, lab) in enumerate(((cor, "corrupted"), (rec, "recovered"),
                                          (clean, "clean"))):
                axes[i, j].imshow(t[j].permute(1, 2, 0).numpy())
                axes[i, j].axis("off")
                if j == 0:
                    axes[i, j].set_title(lab, loc="left", fontsize=9)
        fig.suptitle(f"recovery AE w={w} -- if row 2 is flat grey, it collapsed")
        out = config.FIG_DIR / f"diagnose_recovery_w{w}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        print(f"\nsaved -> {out}")
