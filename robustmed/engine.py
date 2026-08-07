"""Training and evaluation loops."""
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from . import config, data


def set_seed(seed=config.SEED):
    import os
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()


# --- checkpoints -----------------------------------------------------------

def save_ckpt(path, model, optimizer=None, epoch=0, extra=None):
    payload = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_ckpt(path, model, optimizer=None):
    """Returns the epoch to resume from, 0 if there is no checkpoint."""
    if not path.exists():
        return 0
    ck = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    if optimizer is not None and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    print(f"resumed <- {path} (epoch {ck.get('epoch', 0)})")
    return ck.get("epoch", 0)


# --- metrics ---------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader):
    """Both accuracies plus the prediction histogram.

    Balanced accuracy is the headline metric: PneumoniaMNIST is imbalanced, so
    a model collapsed onto the majority class still scores ~0.74 raw while
    sitting at chance (0.50) balanced. Raw accuracy alone hides the exact
    failure this project is about.
    """
    model.eval()
    ys, ps = [], []
    for x01, y in loader:
        ps += model(x01.to(DEVICE)).argmax(1).cpu().tolist()
        ys += y.tolist()
    ys, ps = np.array(ys), np.array(ps)
    return {
        "accuracy": float((ys == ps).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(ys, ps)),
        "n": int(ys.size),
        "pred_counts": {int(k): int(v) for k, v in Counter(ps.tolist()).items()},
        "confusion": confusion_matrix(ys, ps).tolist(),
    }


def collapsed(result, bal_floor=None):
    """True when a run produced nothing usable.

    Two signatures, because the first one alone undercounts. The original test
    was `len(pred_counts) == 1` -- a single predicted class. But s0_w4 in the
    stability sweep predicted two classes and still scored bal=0.5043, i.e. dead
    on arrival, and was reported as healthy. Anything at or below chance is
    collapsed regardless of how many classes it nominally emits.
    """
    floor = config.COLLAPSE_BAL if bal_floor is None else bal_floor
    return (len(result["pred_counts"]) == 1
            or result["balanced_accuracy"] <= floor)


# --- worst-case and corruption-error metrics -------------------------------

def worst_case(bal, conditions):
    """(condition, balanced_accuracy) of the weakest condition in `conditions`.

    The headline metric. Averaging over conditions answers "how does it do on
    average", but the question is whether robustness holds up under EVERY
    condition, which is a floor. The two differ: in the stability sweep the
    mean-to-worst gap ranged from 0.009 to 0.084 across runs, so two modules
    that tie on the mean can be six points apart on the floor.
    """
    present = [c for c in conditions if c in bal]
    if not present:
        return None, float("nan")
    worst = min(present, key=lambda c: bal[c])
    return worst, bal[worst]


def corruption_error(bal, baseline_bal, conditions):
    """mCE and relative mCE, Hendrycks & Dietterich (ICLR 2019) style.

    Error is 1 - balanced_accuracy, normalized by the baseline's error on the
    same condition, then averaged. mCE < 1 means more robust than the baseline.
    A robustness reviewer expects this alongside balanced accuracy, and it is
    what makes our numbers comparable to the corruption-robustness literature.
    """
    ratios = []
    for c in conditions:
        if c not in bal or c not in baseline_bal:
            continue
        denom = 1.0 - baseline_bal[c]
        if denom <= 1e-8:            # baseline is perfect here; ratio undefined
            continue
        ratios.append((1.0 - bal[c]) / denom)
    return float(np.mean(ratios)) if ratios else float("nan")


def unfixable_conditions(bals, conditions, floor=None):
    """Conditions where the BEST arm is still at chance -- nobody can fix them.

    `bals` is an iterable of {condition: balanced_accuracy}, one per arm.

    The earlier version of this function tested the clean-trained baseline
    instead, which had it exactly backwards. A condition where the baseline sits
    at chance is where recovery has the MOST headroom, not the least: on
    PneumoniaMNIST the baseline is at 0.5000 for gaussian_noise at every
    severity, and that is precisely the family recovery lifts to ~0.78.
    Excluding it would have deleted a trained family from the worst-case metric.

    Unfixable can only be decided after the arms have run, so call this at the
    end and use it to report a separate group -- never to pre-filter.
    """
    floor = config.COLLAPSE_BAL if floor is None else floor
    bals = list(bals)
    out = []
    for c in conditions:
        seen = [b[c] for b in bals if c in b]
        if seen and max(seen) <= floor:
            out.append(c)
    return out


@torch.no_grad()
def reconstruction_quality(ae, pair_loader, max_batches=10):
    """PSNR/SSIM of recovered vs clean.

    Report these, but do not select models with them: the perceptual-loss AE
    trades pixel fidelity for classifier-usable structure, so PSNR ranks the
    widths differently from balanced accuracy.
    """
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim

    ae.eval()
    ps, ss = [], []
    for i, batch in enumerate(pair_loader):
        if i >= max_batches:
            break
        cor, clean = batch[0].to(DEVICE), batch[1].to(DEVICE)
        out = ae(cor)
        ps.append(psnr(out, clean).item())
        ss.append(ssim(out, clean).item())
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}


def make_val_probe(raw, corruptions=None, severities=None, n=None):
    """Pre-render a fixed corrupted validation set ONCE, as uint8.

    Model selection needs a validation score after every epoch, and re-rendering
    corruptions through ImageMagick each time would cost more than the training
    step itself. The corrupted set is deterministic given (image, corruption,
    severity), so it is built once and reused for every epoch, width and seed.
    ~12 KB per image: 256 images x 9 conditions is under 60 MB.

    Returns (x_uint8 [N,3,H,W], y [N]).
    """
    from .corruptions import corrupt

    corruptions = corruptions or config.TRAIN_CORRUPTIONS
    severities = severities or config.TRAIN_SEVERITIES
    n = n or config.VAL_PROBE_N

    xs, ys = [], []
    for i in range(min(n, len(raw))):
        img, lab = raw[i]
        for name in corruptions:
            for sev in severities:
                xs.append(torch.from_numpy(corrupt(img, name, sev)))
                ys.append(data.label_of(lab))
    x = torch.stack(xs).permute(0, 3, 1, 2).contiguous()
    print(f"val probe: {x.shape[0]} images "
          f"({len(corruptions)} corruptions x {len(severities)} severities)")
    return x, torch.tensor(ys)


@torch.no_grad()
def probe_balanced(ae, clf, x_uint8, y, batch_size=256):
    """Balanced accuracy of (ae -> clf) on the cached probe set.

    Balanced accuracy specifically: a collapsed AE still scores near the majority
    rate on raw accuracy, which is exactly the signal we need selection to reject.
    """
    ae.eval()
    preds = []
    for i in range(0, len(y), batch_size):
        xb = x_uint8[i:i + batch_size].to(DEVICE).float() / 255.0
        preds += clf(ae(xb)).argmax(1).cpu().tolist()
    return float(balanced_accuracy_score(y.numpy(), np.array(preds)))


@torch.no_grad()
def latency_ms(model, batch_size=1, iters=200, warmup=20):
    """Per-image latency using CUDA events with explicit syncs.

    The original used time.time() around unsynchronized GPU calls, so it was
    partly measuring kernel launch overhead -- which matters when the claim is
    that recovery is compute-bounded.
    """
    model.eval()
    x = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=DEVICE)
    for _ in range(warmup):
        model(x)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            model(x)
        end.record()
        torch.cuda.synchronize()
        total = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        total = (time.perf_counter() - t0) * 1e3

    return total / iters / batch_size


# --- training --------------------------------------------------------------

def train_classifier(model, train_loader, val_loader, ckpt_path,
                     epochs=config.CLF_EPOCHS, lr=config.CLF_LR):
    """Used for both baselines. The only difference between Baseline 1 and 2 is
    whether train_loader applies MedMNIST-C augmentation."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    start = load_ckpt(ckpt_path, model, opt)

    for ep in range(start, epochs):
        model.train()
        for x01, y in train_loader:
            x01, y = x01.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x01), y).backward()
            opt.step()
        m = evaluate(model, val_loader)
        print(f"epoch {ep+1}/{epochs}  val_acc={m['accuracy']:.4f}  "
              f"val_bal={m['balanced_accuracy']:.4f}")
        save_ckpt(ckpt_path, model, opt, ep + 1)
    print(f"saved -> {ckpt_path}")
    return model


def train_recovery(width, frozen_clf, pair_loader, epochs=config.AE_EPOCHS,
                   lr=config.AE_LR, lambda_max=config.AE_LAMBDA_MAX,
                   warmup=config.AE_WARMUP, residual=None,
                   val_probe=None, seed=None, tag=None,
                   arch="convae", output=None):
    """Perceptual-loss recovery module: MSE(recon, clean) + lam * CE(clf(recon), y).

    lam is 0 for the first `warmup` epochs then ramps to lambda_max. Pure MSE
    (lambda_max=0) converges to the identity map and recovers nothing.

    Pass val_probe=(x_uint8, y) from make_val_probe to enable best-epoch
    selection. This is not a refinement: the CE term has a degenerate minimiser
    (emit whatever the classifier calls the majority class), and a run that falls
    into it late in training used to be saved anyway, because only the final
    epoch was kept. Selection keeps the best epoch by validation balanced
    accuracy instead, and records the full curve so a collapse is visible after
    the fact rather than invisible.
    """
    import copy

    from .models import build_recovery

    ae = build_recovery(arch, width=width, residual=residual, output=output).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    frozen_clf.eval()

    select = val_probe is not None and config.AE_SELECT_BEST
    best = {"bal": -1.0, "state": None, "epoch": 0}
    history = []
    label = f"{arch} w={width}"

    for ep in range(epochs):
        lam = 0.0 if ep < warmup else lambda_max * (ep - warmup + 1) / max(1, epochs - warmup)
        ae.train()
        total, seen, gnorm, batches = 0.0, 0, 0.0, 0
        last_batch = None
        for cor, clean, y in pair_loader:
            cor, clean, y = cor.to(DEVICE), clean.to(DEVICE), y.to(DEVICE)
            out = ae(cor)
            loss = mse(out, clean) + lam * ce(frozen_clf(out), y)
            opt.zero_grad()
            loss.backward()
            # Gradient norm BEFORE the step. A run that is dead reports exactly
            # 0.0 here for every epoch -- that is the signature that separates a
            # saturated-output failure from an ordinary bad run, and without it
            # the two are indistinguishable in the loss curve alone.
            gnorm += float(sum(p.grad.abs().sum() for p in ae.parameters()
                               if p.grad is not None))
            batches += 1
            opt.step()
            total += loss.item() * cor.size(0)
            seen += cor.size(0)
            last_batch = cor

        row = {"epoch": ep + 1, "lam": lam, "loss": total / seen,
               "grad_norm": gnorm / max(1, batches),
               "saturation": ae.saturation(last_batch) if last_batch is not None else float("nan")}
        msg = (f"  [{label}] ep{ep+1}/{epochs} lam={lam:.2f} "
               f"loss={row['loss']:.4f} grad={row['grad_norm']:.2e} "
               f"sat={row['saturation']:.2f}")
        if val_probe is not None:
            row["val_bal"] = probe_balanced(ae, frozen_clf, *val_probe)
            msg += f" val_bal={row['val_bal']:.4f}"
            if select and row["val_bal"] > best["bal"]:
                best = {"bal": row["val_bal"],
                        "state": copy.deepcopy(ae.state_dict()),
                        "epoch": ep + 1}
                msg += " *"
        history.append(row)
        if row["grad_norm"] == 0.0:
            msg += "  <- DEAD (zero gradient)"
        print(msg, flush=True)

    if select and best["state"] is not None:
        ae.load_state_dict(best["state"])
        print(f"  [{label}] selected epoch {best['epoch']} "
              f"(val_bal={best['bal']:.4f}) of {epochs}")

    path = config.ae_ckpt(width, seed=seed, tag=tag, arch=arch)
    torch.save({"model": ae.state_dict(), "width": width, "arch": arch,
                "output": ae.output, "residual": ae.output == "residual",
                "seed": seed, "tag": tag, "lambda_max": lambda_max,
                "selected_epoch": best["epoch"] if select else epochs,
                "history": history}, path)
    print(f"saved -> {path}")
    ae.eval()
    return ae


def load_recovery(width, seed=None, tag=None, arch=None):
    from .models import build_recovery
    ck = torch.load(config.ae_ckpt(width, seed=seed, tag=tag, arch=arch),
                    map_location=DEVICE)
    ae = build_recovery(ck.get("arch", "convae"), width=ck["width"],
                        output=ck.get("output")).to(DEVICE)
    ae.load_state_dict(ck["model"])
    ae.eval()
    return ae


# --- pipeline evaluation ---------------------------------------------------

class Pipeline(nn.Module):
    """recovery -> frozen classifier. recovery=None is the baseline path."""

    def __init__(self, recovery, classifier):
        super().__init__()
        self.recovery, self.classifier = recovery, classifier

    def forward(self, x01):
        if self.recovery is not None:
            x01 = self.recovery(x01)
        return self.classifier(x01)


def eval_conditions(recovery, classifier, raw, conditions=None):
    """Evaluate one recovery+classifier combination across every condition.

    Returns {condition: metrics}. This one function replaces the six
    near-identical eval_* variants in the original notebook.
    """
    conditions = conditions or data.eval_conditions()
    pipe = Pipeline(recovery, classifier)
    out = {}
    for cond in conditions:
        m = evaluate(pipe, data.condition_loader(raw, cond))
        out[cond] = m
        flag = "  <- collapsed" if collapsed(m) else ""
        print(f"  {cond:26s} bal={m['balanced_accuracy']:.4f} "
              f"acc={m['accuracy']:.4f}{flag}")
    return out


def eval_compound(recovery, classifier, raw, pairs=None):
    pairs = pairs or config.COMPOUND_PAIRS
    pipe = Pipeline(recovery, classifier)
    out = {}
    for n1, s1, n2, s2 in pairs:
        loader = data.loader(data.Compound(raw, n1, s1, n2, s2))
        out[f"{n1}+{n2}"] = evaluate(pipe, loader)
    return out
