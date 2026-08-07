"""Cost metrics: what does putting this module in front of the classifier buy you?

Accuracy alone cannot answer the research question. "Efficient under a
constrained environment" is a claim about cost, and cost has several axes that
disagree with each other -- a module can be tiny in parameters and slow in
practice (depthwise convs, many small kernels), or large and fast (one big
matmul). Reporting parameters alone hides that.

The headline is deliberately RELATIVE: every cost is also expressed as a
fraction of the frozen classifier's own cost. "Recovery adds 4% to end-to-end
inference" survives a change of operating point, hardware or backbone; "1,119
parameters" does not, and it anchored this project to a number that later turned
out to be the wrong axis entirely.

CPU single-thread numbers are not a smoke test here -- they are the
constrained-hardware measurement, and the one a clinical-deployment reviewer
will actually care about.
"""
import time

import numpy as np
import torch

from . import config


@torch.no_grad()
def macs(model, image_size=None, batch=1):
    """Multiply-accumulates for one forward pass, via torch's FlopCounterMode.

    Returns None if unavailable rather than guessing: a wrong FLOP count is
    worse than a missing one, because it looks authoritative in a table.
    """
    if model is None:
        return 0
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    px = image_size or config.IMAGE_SIZE
    x = torch.randn(batch, 3, px, px, device=next(model.parameters(),
                                                  torch.zeros(1)).device
                    if any(True for _ in model.parameters()) else "cpu")
    counter = FlopCounterMode(display=False)
    try:
        with counter:
            model(x)
        return counter.get_total_flops() // 2      # FLOPs -> MACs
    except Exception:
        return None


@torch.no_grad()
def latency_ms(model, device, batch=1, iters=50, warmup=10, threads=None):
    """Per-image latency. CUDA events on GPU, perf_counter with syncs on CPU.

    `threads=1` gives the single-thread CPU figure, which is the number that
    matters for edge deployment and the one most papers omit.
    """
    px = config.IMAGE_SIZE
    prev = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        x = torch.randn(batch, 3, px, px, device=device)
        run = (lambda: x) if model is None else (lambda: model(x))
        for _ in range(warmup):
            run()
        if device.type == "cuda":
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                run()
            e.record()
            torch.cuda.synchronize()
            total = s.elapsed_time(e)
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                run()
            total = (time.perf_counter() - t0) * 1e3
        return total / iters / batch
    finally:
        torch.set_num_threads(prev)


@torch.no_grad()
def peak_memory_mb(model, device, batch=1):
    """Peak activation memory for one forward pass. CUDA only; None on CPU."""
    if device.type != "cuda" or model is None:
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def disk_size_mb(model):
    """On-disk footprint at fp32 and fp16 -- what actually ships to a device."""
    if model is None:
        return {"fp32": 0.0, "fp16": 0.0}
    n = sum(p.numel() for p in model.parameters())
    n += sum(b.numel() for b in model.buffers())
    return {"fp32": n * 4 / 1e6, "fp16": n * 2 / 1e6}


def profile(model, classifier=None, device=None, batches=(1, 128)):
    """Every cost axis for one module, plus its share of end-to-end inference.

    `classifier` is the frozen backbone. Passing it turns absolute costs into
    the relative ones the paper should lead with.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_params = 0 if model is None else sum(p.numel() for p in model.parameters())

    out = {
        "params": n_params,
        "macs": macs(model),
        "disk_mb": disk_size_mb(model),
        "peak_mem_mb": peak_memory_mb(model, device, batch=1),
        "latency_ms": {},
        "throughput_img_s": {},
    }
    for b in batches:
        out["latency_ms"][f"bs{b}"] = latency_ms(model, device, batch=b)
        per_img = out["latency_ms"][f"bs{b}"]
        out["throughput_img_s"][f"bs{b}"] = 1e3 / per_img if per_img else None
    if device.type != "cpu":
        out["latency_ms"]["cpu_1thread_bs1"] = latency_ms(
            model, torch.device("cpu"),
            batch=1, iters=20, warmup=5, threads=1)

    if classifier is not None:
        c_params = sum(p.numel() for p in classifier.parameters())
        c_macs = macs(classifier)
        c_lat = latency_ms(classifier, device, batch=1)
        out["vs_classifier"] = {
            "params_pct": 100.0 * n_params / c_params if c_params else None,
            "macs_pct": (100.0 * out["macs"] / c_macs
                         if out["macs"] and c_macs else None),
            # The number to lead with: what recovery adds to end-to-end latency.
            "latency_overhead_pct": (100.0 * out["latency_ms"]["bs1"] / c_lat
                                     if c_lat else None),
        }
    return out
