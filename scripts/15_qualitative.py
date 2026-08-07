"""Qualitative figures from the real scans: what the corruption does, what recovery restores.

Numbers say recovery works; these say what it is doing. Three figures, all built
from actual dataset images at the run's resolution, with the measured balanced
accuracy printed on the panel it belongs to so a reader cannot separate the
picture from the evidence.

  fig_atlas        one real scan under every corruption family in the registry.
                   This is the fault model, shown rather than described -- and it
                   makes the category taxonomy obvious at a glance: noise, blur
                   and codec damage local structure, photometric shifts the whole
                   intensity range without touching structure at all.

  fig_recovery     corrupted / recovered / clean for the conditions that matter,
                   annotated with baseline -> recovered balanced accuracy. The
                   rows are chosen to span categories, not to flatter the model:
                   the photometric row is included precisely because it is the
                   weakest.

  fig_severity     one family across all five severities, corrupted above and
                   recovered below, to show where recovery stops holding.

Run per dataset; --dataset swaps the corpus and the figures land in that
corpus's own directory, so a chest X-ray panel and an ultrasound panel can sit
side by side in a slide.

    python scripts/15_qualitative.py
    python scripts/15_qualitative.py --dataset breastmnist
    python scripts/15_qualitative.py --family gaussian_blur --index 3
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (balanced, config, corruptions, data, engine, models,
                     parse_args, setup, store, frozen_baseline1)


def _show(ax, chw, title=None, sub=None, color=None):
    ax.imshow(np.clip(chw.detach().cpu().permute(1, 2, 0).numpy(), 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#c8d0d8"); s.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=8.5, pad=3, color=color or "#141a20")
    if sub:
        ax.set_xlabel(sub, fontsize=7.5, labelpad=2, color=color or "#5d6b78")


def _tensor(raw, i, family=None, severity=0):
    img, _ = raw[i]
    if family is None:
        return data.to_tensor01(img)
    return data.to_tensor01(corruptions.corrupt(img, family, severity))


if __name__ == "__main__":
    args = parse_args(
        index={"type": int, "default": 0, "help": "which test image to display"},
        severity={"type": int, "default": 4},
        family={"default": None, "help": "family for the severity sweep"},
        arch={"default": "convae"},
        seed={"type": int, "default": 0},
        tag={"default": None},
    )
    train, val, test, info, n_classes = setup(args)
    clf = frozen_baseline1(n_classes)

    tag = args.tag or f"lam{config.AE_LAMBDA_MAX:g}_{config.RECOVERY_OUTPUT}"
    try:
        ae = engine.load_recovery(None, seed=args.seed, tag=tag, arch=args.arch)
        print(f"recovery: {args.arch} published, seed {args.seed}, "
              f"{models.count_params(ae):,} params")
    except FileNotFoundError:
        raise SystemExit(f"no {args.arch} checkpoint -- run 13_module_comparison "
                         f"--published first")

    families = corruptions.names()
    # Measured accuracies, so every panel caption is evidence rather than decoration.
    bl = rec = None
    raw_res = store.load_optional("module_comparison")
    if raw_res:
        bl = balanced(raw_res["baseline1"])
        key = next((k for k in raw_res if k.startswith(args.arch)), None)
        rec = balanced(raw_res[key]) if key else None
        if key:
            print(f"annotating with measured accuracy from {key}")

    def acc(cond):
        if not bl:
            return None
        a = bl.get(cond)
        b = rec.get(cond) if rec else None
        if a is None:
            return None
        return f"{a:.2f}" + (f" → {b:.2f}" if b is not None else "")

    px = config.IMAGE_SIZE
    print(f"\nbuilding figures at {px}px from {config.DATA_FLAG} test image "
          f"#{args.index}")

    # --- 1. the fault model, shown -----------------------------------------
    cols = 5
    rows = int(np.ceil((len(families) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.05 * cols, 2.35 * rows))
    axes = np.atleast_2d(axes).ravel()
    _show(axes[0], _tensor(test, args.index), "clean", acc(data.CLEAN))
    for k, fam in enumerate(families, start=1):
        cond = data.condition(fam, args.severity)
        _show(axes[k], _tensor(test, args.index, fam, args.severity),
              fam.replace("_", " "), acc(cond))
    for ax in axes[len(families) + 1:]:
        ax.axis("off")
    fig.suptitle(f"{config.DATA_FLAG} — every corruption family at severity "
                 f"{args.severity + 1}/5   ({px}px)\n"
                 f"captions: balanced accuracy, no recovery → with recovery",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    print("saved ->", _p := config.FIG_DIR / "qual1_corruption_atlas.png")
    fig.savefig(_p, dpi=170, bbox_inches="tight"); plt.close(fig)

    # --- 2. corrupted / recovered / clean ----------------------------------
    # One family per category, so the panel spans the failure modes rather than
    # the ones the module happens to handle well.
    picks, seen_cat = [], set()
    for fam in families:
        cat = config.category_of(fam)
        if cat not in seen_cat:
            picks.append((cat, fam)); seen_cat.add(cat)
    fig, axes = plt.subplots(len(picks), 3, figsize=(6.6, 2.3 * len(picks)))
    axes = np.atleast_2d(axes)
    clean_t = _tensor(test, args.index)
    for r, (cat, fam) in enumerate(picks):
        cond = data.condition(fam, args.severity)
        cor = _tensor(test, args.index, fam, args.severity)
        with torch.no_grad():
            out = ae(cor.unsqueeze(0).to(engine.DEVICE))[0].cpu()
        a = bl.get(cond) if bl else None
        b = rec.get(cond) if rec else None
        _show(axes[r, 0], cor, f"{fam.replace('_',' ')}  ·  {cat}",
              None if a is None else f"no recovery  {a:.3f}")
        _show(axes[r, 1], out, "recovered",
              None if b is None else f"with recovery  {b:.3f}",
              color="#0b7a8c")
        _show(axes[r, 2], clean_t, "clean reference", None)
    fig.suptitle(f"{config.DATA_FLAG} — one family per corruption category, "
                 f"severity {args.severity + 1}/5\ncaptions: balanced accuracy "
                 f"over the whole test split", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    print("saved ->", _p := config.FIG_DIR / "qual2_recovery.png")
    fig.savefig(_p, dpi=170, bbox_inches="tight"); plt.close(fig)

    # --- 3. where it stops holding -----------------------------------------
    # Fall back by CATEGORY, not by list order. gaussian_blur is absent from 6
    # of the 12 registries, and taking families[0] there lands on whatever
    # happens to sort first -- on ultrasound that is brightness_down, a
    # photometric family, which is the wrong thing to sweep when the question is
    # how far spatial recovery holds.
    fam = args.family
    if fam not in families:
        blurs = [f for f in families if config.category_of(f) == "blur"]
        fam = blurs[0] if blurs else families[0]
    sevs = list(range(config.N_SEVERITIES))
    fig, axes = plt.subplots(2, len(sevs), figsize=(2.05 * len(sevs), 4.9))
    for j, s in enumerate(sevs):
        cond = data.condition(fam, s)
        cor = _tensor(test, args.index, fam, s)
        with torch.no_grad():
            out = ae(cor.unsqueeze(0).to(engine.DEVICE))[0].cpu()
        a = bl.get(cond) if bl else None
        b = rec.get(cond) if rec else None
        _show(axes[0, j], cor, f"severity {s + 1}",
              None if a is None else f"{a:.3f}")
        _show(axes[1, j], out, None, None if b is None else f"{b:.3f}",
              color="#0b7a8c")
    axes[0, 0].set_ylabel("corrupted", fontsize=9)
    axes[1, 0].set_ylabel("recovered", fontsize=9)
    fig.suptitle(f"{config.DATA_FLAG} — {fam.replace('_',' ')} across all five "
                 f"severities\ncaptions: balanced accuracy", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    print("saved ->", _p := config.FIG_DIR / f"qual3_severity_{fam}.png")
    fig.savefig(_p, dpi=170, bbox_inches="tight"); plt.close(fig)

    print(f"\nall three in {config.FIG_DIR}")
