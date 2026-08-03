"""Paper figures. Each reads from results/ and writes a PNG to figures/.

Row labels use set_ylabel with ticks removed rather than axis("off") --
axis("off") hides the label, which is why the row captions never appeared in
the original Figs 2 and 3.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import config, data

RED = "#C62828"      # no recovery / Baseline 1
BLUE = "#1565C0"     # our recovery module
GREEN = "#2E7D32"    # Baseline 2 / augmentation
GRAY = "#757575"


def _save(fig, name):
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved -> {path}")
    plt.close(fig)
    return path


def _bare(ax, label=None):
    """Image axis with no ticks but a usable ylabel."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if label:
        ax.set_ylabel(label, fontsize=9)


def fig_corruption_grid(clean_pil, corruptions=None, severities=None,
                        name="fig1_corruptions.png"):
    """Fig 1 -- what the corruptions look like at each severity."""
    from .corruptions import corrupt

    corruptions = corruptions or config.EVAL_CORRUPTIONS
    severities = severities or config.EVAL_SEVERITIES
    fig, axes = plt.subplots(len(corruptions), len(severities) + 1,
                             figsize=(2.1 * (len(severities) + 1), 2.1 * len(corruptions)))
    for r, c in enumerate(corruptions):
        _bare(axes[r, 0], c.replace("_", "\n"))
        axes[r, 0].imshow(np.asarray(clean_pil))
        if r == 0:
            axes[r, 0].set_title("clean", fontsize=10)
        for k, s in enumerate(severities):
            _bare(axes[r, k + 1])
            axes[r, k + 1].imshow(corrupt(clean_pil, c, s))
            if r == 0:
                axes[r, k + 1].set_title(f"severity {s + 1}", fontsize=10)
    fig.suptitle(f"MedMNIST-C corruptions on PneumoniaMNIST "
                 f"({config.IMAGE_SIZE}x{config.IMAGE_SIZE})", fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def fig_collapse_recovery(baseline1, baseline2, recovery, conditions=None,
                          recovery_label=None, name="fig2_collapse_recovery.png"):
    """Fig 2 -- the headline: collapse, and how much each method recovers.

    Each argument is {condition: balanced_accuracy}. Pass baseline2=None to
    render before the augmentation baseline has been run.
    """
    conditions = conditions or data.eval_conditions()
    x = np.arange(len(conditions))
    series = [("Baseline 1 (clean-trained, frozen)", baseline1, RED),
              (recovery_label or "+ Recovery AE (ours)", recovery, BLUE)]
    if baseline2:
        series.append(("Baseline 2 (aug-trained, Paper A)", baseline2, GREEN))

    w = 0.8 / len(series)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * w

    fig, ax = plt.subplots(figsize=(1.05 * len(conditions) + 3, 5))
    for (label, vals, color), off in zip(series, offsets):
        ax.bar(x + off, [vals[c] for c in conditions], w, label=label, color=color)
    ax.axhline(0.5, ls=":", c=GRAY, label="chance (balanced)")
    ax.set_xticks(x)
    ax.set_xticklabels([data.pretty(c) for c in conditions], fontsize=8)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Collapse and recovery in front of a frozen classifier")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, name)


def fig_reconstructions(corrupted, recovered, clean, width, n=6,
                        name="fig3_reconstructions.png"):
    """Fig 3 -- corrupted / recovered / clean triplets. Tensors are CHW in [0,1]."""
    n = min(n, corrupted.shape[0])
    rows = [("corrupted", corrupted), (f"recovered (w={width})", recovered),
            ("clean target", clean)]
    fig, axes = plt.subplots(3, n, figsize=(2.1 * n, 6.6))
    for r, (label, batch) in enumerate(rows):
        for j in range(n):
            _bare(axes[r, j], label if j == 0 else None)
            axes[r, j].imshow(batch[j].detach().cpu().permute(1, 2, 0).numpy())
    fig.suptitle("Recovery autoencoder reconstructions", fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def fig_pareto(widths, params, gains, latencies=None, clean_preservation=None,
               name="fig4_pareto.png"):
    """Fig 4 (main RQ) -- recovery gain vs cost.

    Colour encodes clean-accuracy preservation, so the reader sees the
    accuracy/cost trade-off and the clean-input regression in one panel.
    """
    ncols = 2 if latencies else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5), squeeze=False)

    def panel(ax, xs, xlabel, logx):
        if clean_preservation:
            sc = ax.scatter(xs, gains, s=140, c=clean_preservation, cmap="viridis",
                            edgecolor="k", zorder=3)
            fig.colorbar(sc, ax=ax, label="clean-accuracy preservation")
        else:
            ax.scatter(xs, gains, s=140, color=BLUE, edgecolor="k", zorder=3)
        ax.plot(xs, gains, "--", color=GRAY, zorder=2)
        for wd, xp, yp in zip(widths, xs, gains):
            ax.annotate(f"w={wd}", (xp, yp), textcoords="offset points",
                        xytext=(8, -4), fontsize=8)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean recovery gain (balanced acc)")

    panel(axes[0, 0], params, "recovery module parameters (log)", True)
    axes[0, 0].set_title("Recovery gain vs model size")
    if latencies:
        panel(axes[0, 1], latencies, "inference latency (ms/image)", False)
        axes[0, 1].set_title("Recovery gain vs latency")
    fig.tight_layout()
    return _save(fig, name)


def fig_generalization(compound=None, transfer=None, name="fig5_generalization.png"):
    """Fig 5 -- unseen compound corruptions and cross-domain transfer.

    Both panels are optional; whichever experiments exist get drawn.
    """
    panels = [(t, d) for t, d in (("Unseen compound corruptions", compound),
                                  ("Kermany cross-domain transfer", transfer)) if d]
    if not panels:
        print("nothing to plot -- run the compound or transfer stage first")
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6),
                             squeeze=False)
    for ax, (title, rows) in zip(axes[0], panels):
        labels = list(rows.keys())
        no_rec = [rows[k]["no_recovery"] for k in labels]
        with_rec = [rows[k]["with_recovery"] for k in labels]
        x, w = np.arange(len(labels)), 0.38
        ax.bar(x - w / 2, no_rec, w, label="no recovery", color=RED)
        ax.bar(x + w / 2, with_rec, w, label="with recovery", color=BLUE)
        ax.axhline(0.5, ls=":", c=GRAY)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("gaussian_", "g_").replace("jpeg_compression", "jpeg")
                            for l in labels], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("balanced accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Generalization beyond the training corruptions", fontsize=12)
    fig.tight_layout()
    return _save(fig, name)
