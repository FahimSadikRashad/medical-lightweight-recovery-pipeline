"""Stage 2b: one skeleton, one budget, one mechanism at a time.

Stage 2a compared whole architectures at their published sizes -- 14k to 29M,
different depths, different training stability. Two of the five never completed
a CE-active run at all, so their rows say nothing about their mechanisms. That
comparison cannot answer "which concept should K3 inherit", because every
difference is confounded with depth, width and optimisability.

This file removes those confounds. Every variant is the SAME network -- stem,
N blocks, head, the shared bounded output -- with exactly one mechanism switched
on, and its width tuned so all variants land on the same parameter budget. The
only thing that differs is the mechanism.

It also rescues the two arms Stage 2a could not measure. SPAN's attention and
NAFNet's U-Net diverged at 386k and 29M on 4,708 images; at ~14k they are inside
the regime where everything else trains, so the mechanism finally gets a fair
test even though the published architecture did not.

    S0  plain              3x3 conv + ReLU                     (control)
    S1  multiscale         stride-2 down / up                  ConvAE, NAFNet
    S2  pf_attention       U * (sigmoid(H) - 0.5)              SPAN
    S3  channel_attention  global avg pool -> 1x1 -> multiply   NAFNet SCA
    S4  safm               multi-level pooling modulation       SAFMN
    S5  simple_gate        channel split -> multiply            NAFNet

Read S3 vs S4 first. Photometric corruption is a global intensity remap and the
binding category; S3 is the only mechanism with a purely global receptive field,
S4 is global-ish plus spatially selective. Stage 2a suggested S4 wins that
contest, and this is the controlled test of it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import RecoveryModule

MECHANISMS = ("plain", "multiscale", "pf_attention", "channel_attention",
              "safm", "simple_gate")


class _Block(nn.Module):
    """One conv block carrying at most one mechanism."""

    def __init__(self, c, mechanism):
        super().__init__()
        self.mechanism = mechanism
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

        if mechanism == "channel_attention":
            self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        elif mechanism == "safm":
            levels = 4
            while levels > 1 and c % levels:
                levels -= 1
            self.levels, self.split = levels, c // levels
            self.dw = nn.ModuleList([
                nn.Conv2d(self.split, self.split, 3, padding=1, groups=self.split)
                for _ in range(levels)])
            self.proj = nn.Conv2d(c, c, 1)
        elif mechanism == "simple_gate":
            self.expand = nn.Conv2d(c, c * 2, 1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        m = self.mechanism

        if m == "pf_attention":
            # SPAN: residual added BEFORE attention, activation symmetric about 0
            return (x + h) * (torch.sigmoid(h) - 0.5)
        if m == "channel_attention":
            return x + h * self.sca(h)
        if m == "safm":
            parts = torch.split(h, self.split, dim=1)[:self.levels]
            hh, ww = h.shape[-2:]
            outs = []
            for i, (p, conv) in enumerate(zip(parts, self.dw)):
                if i == 0:
                    outs.append(conv(p))
                else:
                    s = 2 ** i
                    q = conv(F.adaptive_max_pool2d(p, (max(1, hh // s), max(1, ww // s))))
                    outs.append(F.interpolate(q, size=(hh, ww), mode="nearest"))
            return x + h * torch.sigmoid(self.proj(torch.cat(outs, dim=1)))
        if m == "simple_gate":
            a, b = self.expand(h).chunk(2, dim=1)
            return x + a * b
        return x + h                                        # plain, multiscale


class MechanismNet(RecoveryModule):
    """The Stage 2b skeleton. `mechanism` is the only thing that varies."""

    def __init__(self, width=16, mechanism="plain", blocks=2, output=None, **kw):
        if mechanism not in MECHANISMS:
            raise KeyError(f"unknown mechanism {mechanism!r}; have {MECHANISMS}")
        super().__init__(output=output)
        self.width, self.mechanism = width, mechanism
        self.multiscale = mechanism == "multiscale"

        self.stem = nn.Conv2d(3, width, 3, padding=1)
        if self.multiscale:
            self.down = nn.Conv2d(width, width, 3, stride=2, padding=1)
            self.up = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.blocks = nn.Sequential(*[_Block(width, mechanism)
                                      for _ in range(blocks)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    def body(self, x):
        z = self.stem(x)
        if self.multiscale:
            h, w = z.shape[-2:]
            z = self.up(self.blocks(self.down(z)))[:, :, :h, :w]
        else:
            z = self.blocks(z)
        return self.head(z)


def fit_macs(mechanism, target_macs, blocks=2, lo=2, hi=256):
    """Largest width whose MACs stay within budget.

    Prefer this to fit_width for anything the paper calls an efficiency claim.
    At 224px, matching parameters does NOT match cost: convae at 14,067 params
    costs 55 MMACs and 1.2 ms on one CPU thread, while a same-resolution
    mechanism variant at 13,773 params costs 674 MMACs and 38 ms -- 12x the
    compute and 33x the latency for the same parameter count, because convae
    downsamples twice and the others do not. Matching parameters would hand the
    same-resolution mechanisms a 12x compute advantage and call it a fair test.
    """
    from . import efficiency

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        m = MechanismNet(width=mid, mechanism=mechanism, blocks=blocks)
        n = efficiency.macs(m)
        if n is None:
            raise RuntimeError("FlopCounterMode unavailable; cannot match MACs")
        if n <= target_macs:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def fit_width(mechanism, target_params, blocks=2, lo=2, hi=256):
    """Largest width whose parameter count stays within `target_params`.

    Matching the budget is the whole point of this stage: mechanisms cost
    different amounts per channel, so holding WIDTH fixed would silently hand
    the cheap ones more effective capacity and turn a mechanism comparison back
    into a capacity comparison.

    NOTE: parameters are the wrong budget for an efficiency claim -- see
    fit_macs. Use this only when the question is genuinely about capacity.
    """
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        n = sum(p.numel() for p in
                MechanismNet(width=mid, mechanism=mechanism, blocks=blocks).parameters())
        if n <= target_params:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best
