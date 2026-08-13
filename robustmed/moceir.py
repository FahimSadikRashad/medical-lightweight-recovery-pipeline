"""MoCE-IR: mixture-of-complexity-experts restoration, ported as a stretch arm.

Zamfir et al., CVPR 2025, arXiv:2411.18466 -- "Complexity Experts are
Task-Discriminative Learners for Any Image Restoration." Continues the
routing-inspiration note in docs/RELATED_WORK.md's "cost-aware routing is
already published" section: this is that mechanism, ported in rather than
described secondhand, so this project's own routing claims (if any) have to
be stated as a difference from THIS, not from a summary of it.

Architecture, per the paper (arXiv:2411.18466v1) and its equations 1-7:

    backbone     Restormer-style U-Net. 3x3 stem, 4 encoder levels (depths
                 4,6,6,8; level 4 is the bottleneck, no downsample after it),
                 3 decoder levels (depths 2,4,4) with additive skips, channel
                 width doubling per encoder level from `width`.
    encoder blk  norm -> MDTA (channel/'transposed' self-attention, linear in
                 image size) -> residual, norm -> GDFN (gated-dconv FFN) -> residual.
    decoder blk  same, but MDTA is replaced by a MoCELayer (below).

    MoCELayer    n=4 nested "complexity experts", each a windowed
                 self-attention at channel dim r_i = C/2^i and window size
                 w_i = 2^(1+i) for i in 1..n, i.e. 4/8/16/32 (Eq. 2 uses
                 2^(2+i) = 8/16/32/64; halved here -- see the deviations list,
                 this is the direct consequence of dropping FFT-WSA) --
                 narrower channels and a
                 SMALLER window make the early experts cheap and local; later
                 experts get more channels and a bigger window, i.e. more
                 receptive field, at higher cost. Plus one SHARED expert (an
                 MDTA over the whole feature map) whose output modulates
                 whichever complexity expert was selected, by elementwise
                 multiply (Eq. 1): y = complexity_expert(x) * shared(x).

    router       Top-1 per SAMPLE (not per token), Eq. 4: g(x) =
                 top1(softmax(Wx + eps)), eps ~ N(0, 1/n^2) added during
                 training only. Only the chosen expert actually runs for each
                 sample -- that dispatch pattern IS "irrelevant experts
                 bypassed at inference" (Eq. 3's claim); it is not a separate
                 optimization on top of top-1 routing, it falls out of it.

    cost bias    b_i = p_i / max(p), the "spring force" of the paper's title
                 (Eq. 6) -- a fixed, parameter-count-derived bias that enters
                 the auxiliary load-balancing loss (Eq. 5, 7), pulling routing
                 toward the cheaper expert unless the harder degradation
                 outweighs it.

Deviations from the official implementation, named rather than taken silently
(this file has NOT had arms.py's numeric parameter-count verification pass --
do that before citing an exact match to the paper's 11.47M/25.35M):

  - Each expert's attention is a direct windowed self-attention, not the
    paper's FFT-accelerated version ("FFT-WSA"). Same representational
    mechanism (attention restricted to a window, at a shrunk channel width);
    the FFT is a speed trick this port does not reproduce, so its FLOPs
    number will not match the paper's even where the parameter count does.
    Direct attention cost is O(window^4) per window (an N=window^2 token
    attention matrix), so the paper's literal 8/16/32/64 progression OOM'd
    at 224px in testing -- the largest window, applied at near-full decoder
    resolution with many windows per image, produced multi-gigabyte attention
    tensors per batch. Window sizes are halved to 4/8/16/32 here specifically
    to stay tractable without the FFT accelerator; this is a direct,
    load-bearing consequence of the deviation above, not an independent choice.
  - Even after halving the windows: measured 7.7GB peak RSS for batch=2,
    224px, one forward+backward pass, on CPU. This project's default
    BATCH_SIZE is 128 -- running this arch at that batch size WILL exhaust
    memory. scripts/16_classifier_free_restoration.py therefore excludes
    "moceir" from its default --arms and expects a much smaller --batch-size
    (4-8) when it is requested explicitly; do not add it to any sweep that
    assumes the project's usual batch size without checking this first.
  - The auxiliary loss (Eq. 5-7) is COMPUTED (MoCEIR.aux_loss, accumulated
    across every decoder block after a forward pass) but NOT wired into any
    training loop in this codebase -- engine.train_restoration does not add
    it. Routing therefore currently trains on the reconstruction signal
    alone. The paper's own motivation predicts this biases routing away from
    the cost-aware behaviour that is this module's whole point; wire the aux
    loss in before reporting a claim about WHICH expert gets chosen and why.
  - No skip-connection fusion conv: decoder levels add the upsampled feature
    to its matching encoder skip directly (the same idiom arms.py's NAFNet
    already uses here, `z = up(z) + skip`), not concatenate-then-project.
    Restormer's own decoder does the latter; this halves the extra
    parameters that convention would add and was not something this
    checked against the source for a match.
  - Backbone dimensions (GDFN expansion ratio, head counts per level, no
    separate 'refinement' stage after the last decoder level) are a
    reasonable Restormer-style U-Net, not verified line-by-line against the
    paper's own backbone the way MDTA/GDFN's textbook forms are. Treat the
    MoCELayer mechanism as the verified part of this file and the backbone
    around it as an approximation carrying that mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import RecoveryModule


def window_partition(x, win):
    """BCHW -> (B*num_windows, C, win, win). Requires H, W divisible by win --
    callers pad first (see ComplexityExpert)."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // win, win, w // win, win)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, win, win)
    return windows, (h // win, w // win)


def window_reverse(windows, win, grid, b):
    gh, gw = grid
    c = windows.shape[1]
    x = windows.view(b, gh, gw, c, win, win)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, gh * win, gw * win)


class MDTA(nn.Module):
    """Multi-Dconv-head transposed attention (Restormer, Zamfir et al.'s own
    backbone). Attention computed across CHANNELS, not spatial tokens --
    the CxC map per head is what keeps this linear in image size, unlike
    ordinary self-attention's NxN. Used both as the encoder's plain attention
    and, unmodified, as the MoCELayer's shared expert S(x).

    Returns (out, 0.0) rather than a bare tensor, so TransformerBlock can
    treat every attn module -- MDTA or MoCELayer -- identically.
    """

    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3, bias=False)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)

        def split(t):
            return t.view(b, self.heads, c // self.heads, h * w)

        q, k, v = split(q), split(k), split(v)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        out = (attn @ v).view(b, c, h, w)
        return self.proj(out), 0.0


class GDFN(nn.Module):
    """Gated-Dconv feed-forward network (Restormer): 1x1 expand to 2x hidden,
    depthwise 3x3, split in half, GELU-gate one half by the other, 1x1 project."""

    def __init__(self, c, expand=2.66):
        super().__init__()
        hidden = int(c * expand)
        self.proj_in = nn.Conv2d(c, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.proj_out = nn.Conv2d(hidden, c, 1)

    def forward(self, x):
        x1, x2 = self.dw(self.proj_in(x)).chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    """norm -> attn -> residual, norm -> GDFN -> residual. `attn` is MDTA for
    encoder blocks, a MoCELayer for decoder blocks -- both return (out, aux),
    so this class never has to know which one it holds."""

    def __init__(self, c, attn):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, c)          # LayerNorm2d, same choice arms.py's NAFBlock makes
        self.attn = attn
        self.norm2 = nn.GroupNorm(1, c)
        self.ffn = GDFN(c)
        self.aux_loss = 0.0

    def forward(self, x):
        a, self.aux_loss = self.attn(self.norm1(x))
        x = x + a
        return x + self.ffn(self.norm2(x))


class ComplexityExpert(nn.Module):
    """One nested complexity expert: project C -> r, windowed self-attention
    at window size `window`, project back to C. See the module docstring's
    first deviation for the FFT-WSA vs direct-attention difference."""

    def __init__(self, c, r, window, heads=1):
        super().__init__()
        self.window, self.heads, self.r = window, heads, r
        self.proj_in = nn.Conv2d(c, r, 1)
        self.qkv = nn.Conv2d(r, r * 3, 1, bias=False)
        self.proj_out = nn.Conv2d(r, c, 1)
        self.scale = (r // heads) ** -0.5

    def forward(self, x):
        b, _, h, w = x.shape
        win = self.window
        ph, pw = (-h) % win, (-w) % win
        z = F.pad(x, (0, pw, 0, ph))
        z = self.proj_in(z)
        windows, grid = window_partition(z, win)               # (Bn, r, win, win)
        bn = windows.shape[0]
        n_tok = win * win
        qkv = self.qkv(windows).flatten(2).transpose(1, 2)     # (Bn, N, 3r)
        q, k, v = qkv.chunk(3, dim=-1)

        def split(t):
            return t.view(bn, n_tok, self.heads, self.r // self.heads).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bn, n_tok, self.r)
        out = out.transpose(1, 2).reshape(bn, self.r, win, win)
        out = window_reverse(out, win, grid, b)
        out = self.proj_out(out)
        return out[:, :, :h, :w]


class MoCELayer(nn.Module):
    """See the module docstring for the full mechanism and its deviations.
    `heads` is the block's own head count; each expert gets
    max(1, heads // 2^i), scaled down the same way its channel width is."""

    def __init__(self, c, n_experts=4, heads=4):
        super().__init__()
        self.n_experts = n_experts
        self.shared = MDTA(c, heads=heads)
        self.experts = nn.ModuleList([
            ComplexityExpert(c, r=max(1, c // (2 ** i)), window=2 ** (1 + i),
                             heads=max(1, heads // (2 ** i)))
            for i in range(1, n_experts + 1)])
        p = torch.tensor([sum(w.numel() for w in e.parameters())
                          for e in self.experts], dtype=torch.float32)
        self.register_buffer("cost_bias", p / p.max())         # b_i, Eq. 6
        self.router = nn.Linear(c, n_experts)

    def forward(self, x):
        shared_out, _ = self.shared(x)

        logits = self.router(x.mean((2, 3)))                    # (B, n) -- image-level, Eq. 4
        if self.training:
            logits = logits + torch.randn_like(logits) / (self.n_experts ** 2)
        probs = logits.softmax(dim=-1)
        choice = probs.argmax(dim=-1)                            # top-1

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = choice == i
            if mask.any():                                       # unselected experts never run
                out[mask] = expert(x[mask])

        return out * shared_out, self._aux_loss(probs, choice)

    def _aux_loss(self, probs, choice):
        """Load-balancing + cost-bias auxiliary loss, Eq. 5-7. Computed and
        returned; see the module docstring's second deviation for why it is
        not yet added to any training loss in this codebase."""
        importance = probs.mean(0) * self.cost_bias
        load = torch.bincount(choice, minlength=self.n_experts).float() / choice.numel()
        cv = lambda t: t.std() / (t.mean() + 1e-8)
        return 0.5 * cv(importance) ** 2 + 0.5 * cv(load) ** 2


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c * 2, 3, 2, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c, c // 2, 4, 2, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


class MoCEIR(RecoveryModule):
    """See the module docstring for the verified mechanism and every named
    deviation. `width=None` (this project's usual "published" convention)
    builds the config below -- there is no separate '-S' config in this port;
    the paper's -S/full split is a channel-width choice this file has not
    calibrated to either published parameter count (see the deviations list).
    """

    # Verified by construction: 10,030,679 params -- between the paper's
    # MoCE-IR-S (11.47M) and roughly a third of full MoCE-IR (25.35M), but not
    # calibrated to match either; see the class docstring's deviations.
    PUBLISHED = {"width": 32, "enc_depths": (4, 6, 6, 8), "dec_depths": (2, 4, 4),
                 "n_experts": 4}

    def __init__(self, width=32, enc_depths=(4, 6, 6, 8), dec_depths=(2, 4, 4),
                n_experts=4, output=None, **kw):
        super().__init__(output=output)
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)

        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        c = width
        for depth in enc_depths[:-1]:
            heads = max(1, c // 32)
            self.encoders.append(nn.Sequential(
                *[TransformerBlock(c, MDTA(c, heads)) for _ in range(depth)]))
            self.downs.append(Downsample(c))
            c *= 2
        heads = max(1, c // 32)                                  # bottleneck, no downsample after it
        self.bottleneck = nn.Sequential(
            *[TransformerBlock(c, MDTA(c, heads)) for _ in range(enc_depths[-1])])

        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        for depth in dec_depths:
            self.ups.append(Upsample(c))
            c //= 2
            heads = max(1, c // 32)
            self.decoders.append(nn.Sequential(
                *[TransformerBlock(c, MoCELayer(c, n_experts, heads)) for _ in range(depth)]))
        self.pad = 2 ** len(enc_depths[:-1])
        self.aux_loss = 0.0

    def body(self, x):
        _, _, h, w = x.shape
        ph, pw = (-h) % self.pad, (-w) % self.pad
        z = F.pad(x, (0, pw, 0, ph))

        z = self.stem(z)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            z = enc(z)
            skips.append(z)
            z = down(z)
        z = self.bottleneck(z)

        self.aux_loss = 0.0
        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            z = up(z) + skip
            z = dec(z)
            for blk in dec:
                self.aux_loss = self.aux_loss + blk.aux_loss
        return self.ending(z)[:, :, :h, :w]
