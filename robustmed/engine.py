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


def collapsed(result):
    """True when the model predicts a single class -- the collapse signature."""
    return len(result["pred_counts"]) == 1


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
                   warmup=config.AE_WARMUP, residual=config.AE_RESIDUAL):
    """Perceptual-loss autoencoder: MSE(recon, clean) + lam * CE(clf(recon), y).

    lam is 0 for the first `warmup` epochs then ramps to lambda_max. Pure MSE
    (lambda_max=0) converges to the identity map and recovers nothing.
    """
    from .models import ConvAE

    ae = ConvAE(width, residual=residual).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    frozen_clf.eval()

    for ep in range(epochs):
        lam = 0.0 if ep < warmup else lambda_max * (ep - warmup + 1) / max(1, epochs - warmup)
        ae.train()
        total, seen = 0.0, 0
        for cor, clean, y in pair_loader:
            cor, clean, y = cor.to(DEVICE), clean.to(DEVICE), y.to(DEVICE)
            out = ae(cor)
            loss = mse(out, clean) + lam * ce(frozen_clf(out), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * cor.size(0)
            seen += cor.size(0)
        print(f"  [w={width}] ep{ep+1}/{epochs} lam={lam:.2f} loss={total/seen:.4f}")

    torch.save({"model": ae.state_dict(), "width": width, "residual": residual},
               config.ae_ckpt(width))
    print(f"saved -> {config.ae_ckpt(width)}")
    return ae


def load_recovery(width):
    from .models import ConvAE
    ck = torch.load(config.ae_ckpt(width), map_location=DEVICE)
    ae = ConvAE(ck["width"], residual=ck["residual"]).to(DEVICE)
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
