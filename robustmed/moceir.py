"""MoCE-IR: mixture-of-complexity-experts restoration, ported as a stretch arm.

Zamfir et al., CVPR 2025, arXiv:2411.18466 -- "Complexity Experts are
Task-Discriminative Learners for Any Image Restoration." Continues the
routing-inspiration note in docs/RELATED_WORK.md's "cost-aware routing is
already published" section.

REWRITE NOTE. The first version of this file was built from the paper's prose
and equations alone. Reading the actual source
(github.com/eduardzamfir/MoCE-IR/blob/main/src/net/moce_ir.py) turned up
several real differences, and this rewrite follows the code, not the earlier
guesses:

  - The backbone is 4-5x SHALLOWER than assumed: num_blocks=[1,1,1,3],
    num_dec_blocks=[1,1,1] (9 blocks total), not a Restormer-scale
    enc(4,6,6,8)/dec(2,4,4) guess (42 blocks). Down/up-sampling is
    PixelUnshuffle/PixelShuffle (lossless channel-space reshaping), not
    plain strided conv.
  - Each complexity expert projects down to a small CONSTANT `rank` (default
    2) BEFORE any patch operation, and stays at that width until projecting
    back up. Complexity varies via kernel_size ([3,5,7,9]), patch_size
    ([4,8,16,32]) and per-expert DEPTH (linearly increasing), not primarily
    via channel width as the first port assumed. This is the actual reason
    real MoCE-IR-S is affordable at patch_size=32: the expensive op runs on
    ~2 channels, not tens of channels.
  - "FFTAttention" is not softmax attention. It computes
    rfft2(q) * rfft2(k), inverts, and gates v with the result -- by the
    convolution theorem that is a circular convolution of q and k in the
    spatial domain, computed at O(N log N) via the FFT instead of O(N^2) for
    literal windowed self-attention. THIS is the actual fix for the memory
    blowup the first port hit (7.7GB at batch=2, 224px): that port used
    direct windowed attention because it had not yet found this class.
  - The shared-expert gate is SiLU, not a raw multiply: `body(x) *
    F.silu(proj(shared))`.
  - The router adds a frequency-embedding term to its gate logits (high-pass
    filtered bottleneck features through an MLP) and the auxiliary loss uses
    a proper Normal-CDF-based load term (Shazeer-style), not a plain
    bincount. NEITHER is reproduced here -- the router below is avg-pool +
    linear only, and the aux loss is the simpler CV-based form from the
    paper's own equations 5-7, not the code's exact load term. Named, not
    silently dropped: do not cite this file's router or aux loss as a code
    match, only its backbone and expert mechanism.

Architecture, as rebuilt from source:

    backbone     Restormer-style U-Net: 3x3 stem, 3 encoder levels (depths
                 1,1,1) + a 4th bottleneck level (depth 3), all via MDTA
                 (channel/'transposed' attention, linear in image size);
                 PixelUnshuffle(2)+conv halves resolution/doubles channels
                 between levels. Decoder: 3 levels (depths 1,1,1), MDAB
                 replaced by MoCELayer; PixelShuffle(2)+conv undoes it,
                 additive skip from the matching encoder level.

    MoCELayer    n=4 nested experts (FFTExpert below), each: 1x1 project
                 C -> rank (constant, default 2), `depth_i` blocks of
                 (FFT-mix q,k then gate v) at `kernel_size_i`, 1x1 project
                 rank -> C. kernel_sizes=[3,5,7,9], patch_sizes=[4,8,16,32],
                 depths=[1,2,3,4] (stage_depth=1, linear). Plus one shared
                 MDTA over the full block whose SiLU-gated output multiplies
                 the selected expert's output.

    router       Top-1 per SAMPLE, Eq. 4 form: avg-pool -> linear -> logits,
                 + N(0, 1/n^2) noise during training, softmax, argmax. Only
                 the chosen expert runs per sample -- dispatch IS the
                 "irrelevant experts bypassed" property, not a separate step.

    cost bias    b_i = p_i / max(p) enters the auxiliary loss (paper's
                 spring-force framing); computed and returned, NOT wired into
                 any training loop in this codebase -- same caveat as before.

VERIFIED, not estimated: 3,371,328 params (down from the first port's
10,030,679 -- the real depths are far shallower and the constant tiny rank
removes most of what the first port spent on expert width). At 224px, batch=8
trains (peak 7.47GB RSS); batch=16 does not. That is a genuine ~4x batch
improvement over the first port, which OOM'd even at batch=2 -- the FFT-mix
rewrite is a real fix, not just documentation. Still far short of matching the
paper's own 11.47M for MoCE-IR-S; the router's missing frequency-embedding
term and the simplified aux loss are the most likely places that gap comes
from. Use --batch-size 8, not the earlier guess of 2-4, if this arm is
selected in scripts/13 or scripts/16.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import RecoveryModule


class MDTA(nn.Module):
    """Multi-Dconv-head transposed attention (Restormer). Attention computed
    across CHANNELS, not spatial tokens -- linear in image size. Used as the
    encoder's plain attention and as the MoCELayer's shared expert.

    Returns (out, 0.0) rather than a bare tensor, so TransformerBlock can
    treat every attn module -- MDTA or MoCELayer -- identically.
    """

    def __init__(self, c, heads=1):
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
    """Gated-Dconv feed-forward network (Restormer)."""

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
    """norm -> attn -> residual, norm -> GDFN -> residual."""

    def __init__(self, c, attn):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, c)
        self.attn = attn
        self.norm2 = nn.GroupNorm(1, c)
        self.ffn = GDFN(c)
        self.aux_loss = 0.0

    def forward(self, x):
        a, self.aux_loss = self.attn(self.norm1(x))
        x = x + a
        return x + self.ffn(self.norm2(x))


class FFTExpert(nn.Module):
    """One nested complexity expert -- 'ModExpert' + 'FFTAttention' in the
    real source. Projects C -> a small CONSTANT `rank` first; every
    subsequent op runs on `rank` channels, which is what keeps this cheap
    even at patch_size=32 (see module docstring).

    `depth` sequential (q,k,v) triples, each: FFT(q)*FFT(k) inverted (a
    circular convolution of q,k via the convolution theorem, global receptive
    field at O(N log N)), gating v. Residual per block.
    """

    def __init__(self, c, rank, kernel_size, patch_size, depth):
        super().__init__()
        self.patch_size = patch_size
        self.proj_in = nn.Conv2d(c, rank, 1)
        pad = kernel_size // 2
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "q": nn.Conv2d(rank, rank, kernel_size, padding=pad),
                "k": nn.Conv2d(rank, rank, kernel_size, padding=pad),
                "v": nn.Conv2d(rank, rank, kernel_size, padding=pad),
            }) for _ in range(depth)])
        self.proj_out = nn.Conv2d(rank, c, 1)

    def _fft_mix(self, blk, z):
        b, c, h, w = z.shape
        p = self.patch_size
        ph, pw = (-h) % p, (-w) % p
        zp = F.pad(z, (0, pw, 0, ph))
        hp, wp = zp.shape[-2:]
        q, k, v = blk["q"](zp), blk["k"](zp), blk["v"](zp)

        def to_patches(t):
            t = t.view(b, c, hp // p, p, wp // p, p)
            return t.permute(0, 2, 4, 1, 3, 5).reshape(-1, c, p, p)

        qp, kp, vp = to_patches(q), to_patches(k), to_patches(v)
        mixed = torch.fft.irfft2(torch.fft.rfft2(qp) * torch.fft.rfft2(kp), s=(p, p))
        out = mixed * vp

        gh, gw = hp // p, wp // p
        out = out.view(b, gh, gw, c, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, c, hp, wp)
        return out[:, :, :h, :w]

    def forward(self, x):
        z = self.proj_in(x)
        for blk in self.blocks:
            z = z + self._fft_mix(blk, z)
        return self.proj_out(z)


class MoCELayer(nn.Module):
    """See the module docstring for the mechanism and its named gaps
    (router, aux loss)."""

    def __init__(self, c, n_experts=4, heads=1, rank=2, stage_depth=1):
        super().__init__()
        self.n_experts = n_experts
        self.shared = MDTA(c, heads=heads)
        self.shared_gate = nn.Conv2d(c, c, 1)

        patch_sizes = [2 ** (i + 2) for i in range(n_experts)]      # [4,8,16,32]
        kernel_sizes = [3 + 2 * i for i in range(n_experts)]        # [3,5,7,9]
        depths = [stage_depth + i for i in range(n_experts)]        # linear
        self.experts = nn.ModuleList([
            FFTExpert(c, rank, kernel_sizes[i], patch_sizes[i], depths[i])
            for i in range(n_experts)])

        p = torch.tensor([sum(w.numel() for w in e.parameters())
                          for e in self.experts], dtype=torch.float32)
        self.register_buffer("cost_bias", p / p.max())
        self.router = nn.Linear(c, n_experts)

    def forward(self, x):
        shared_out, _ = self.shared(x)
        gate = F.silu(self.shared_gate(shared_out))

        logits = self.router(x.mean((2, 3)))
        if self.training:
            logits = logits + torch.randn_like(logits) / (self.n_experts ** 2)
        probs = logits.softmax(dim=-1)
        choice = probs.argmax(dim=-1)

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = choice == i
            if mask.any():
                out[mask] = expert(x[mask])

        return out * gate, self._aux_loss(probs, choice)

    def _aux_loss(self, probs, choice):
        importance = probs.mean(0) * self.cost_bias
        load = torch.bincount(choice, minlength=self.n_experts).float() / choice.numel()
        cv = lambda t: t.std() / (t.mean() + 1e-8)
        return 0.5 * cv(importance) ** 2 + 0.5 * cv(load) ** 2


class Downsample(nn.Module):
    """PixelUnshuffle(2): C,H,W -> 4C,H/2,W/2, then 1x1 to 2C -- lossless
    channel-space reshaping, matching the real backbone (not a strided conv)."""

    def __init__(self, c):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.proj = nn.Conv2d(c * 4, c * 2, 1, bias=False)

    def forward(self, x):
        return self.proj(self.unshuffle(x))


class Upsample(nn.Module):
    """1x1 to 2C, then PixelShuffle(2): 2C,H,W -> C/2,2H,2W."""

    def __init__(self, c):
        super().__init__()
        self.proj = nn.Conv2d(c, c * 2, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.proj(x))


class MoCEIR(RecoveryModule):
    """See the module docstring for the verified mechanism and every named
    gap. `width=None` builds the config below.
    """

    PUBLISHED = {"width": 32, "enc_depths": (1, 1, 1, 3), "dec_depths": (1, 1, 1),
                 "n_experts": 4, "rank": 2}

    def __init__(self, width=32, enc_depths=(1, 1, 1, 3), dec_depths=(1, 1, 1),
                n_experts=4, rank=2, output=None, **kw):
        super().__init__(output=output)
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)

        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        c = width
        for depth in enc_depths[:-1]:
            self.encoders.append(nn.Sequential(
                *[TransformerBlock(c, MDTA(c, heads=1)) for _ in range(depth)]))
            self.downs.append(Downsample(c))
            c *= 2
        self.bottleneck = nn.Sequential(
            *[TransformerBlock(c, MDTA(c, heads=1)) for _ in range(enc_depths[-1])])

        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        for depth in dec_depths:
            self.ups.append(Upsample(c))
            c //= 2
            self.decoders.append(nn.Sequential(
                *[TransformerBlock(c, MoCELayer(c, n_experts, heads=1, rank=rank))
                  for _ in range(depth)]))
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
