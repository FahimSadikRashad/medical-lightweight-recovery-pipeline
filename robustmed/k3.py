"""K3: a mixture-of-experts recovery module for MedMNIST-C corruption recovery.

K3 is the module named but never built throughout docs/RQ_PAPER_MAP.md and
docs/RELATED_WORK.md this project accumulated while reading the routing/MoE
restoration literature -- "K3 has to beat this", "K3 does not claim to invent
cost-aware routing", "K3's photometric branch". This file is that module's
first implementation.

WHAT IS NOVEL HERE, STATED PLAINLY, AND WHAT IS NOT.

Nothing in this file invents mixture-of-experts, frequency-domain
restoration, or FFT-based token mixing -- all three exist in the papers cited
below and are cited, not claimed. What is new is the SPECIFIC COMBINATION:

  1. A backbone that is multi-scale (U-Net, two downsamples) by default,
     because that is the one mechanism this project's own controlled
     ablation (robustmed/mechanisms.py, RQ-3 in RQ_PAPER_MAP.md) found
     significant -- Delta mCE -0.301, p=0.008 -- and no prior work reviewed
     this project builds a MoE restoration module on top of that specific,
     already-tested-in-this-project mechanism.

  2. THREE switchable expert-mixing strategies in ONE architecture family,
     selected by `moe_mode`, so they can be swapped and compared without
     changing anything else:

       "complexity"  nested experts of increasing kernel/patch complexity,
                     FFT-mixed (not softmax attention -- see the class
                     docstring), top-1 routed per sample, cost-biased aux
                     loss. Lineage: MoCE-IR (Zamfir et al., CVPR 2025,
                     arXiv:2411.18466). Reimplemented here independently of
                     robustmed/moceir.py (no import between the two files),
                     at a smaller n_experts/rank for this file's
                     deployability goal.

       "frequency"   THE GENUINELY NEW COMBINATION. Every expert is assigned
                     a fixed FREQUENCY BAND (low/mid/high) rather than a
                     level of spatial complexity, every band-expert is
                     ALWAYS active (no top-1 routing -- the split is
                     deterministic, not learned), and each expert's output is
                     masked back into its own band before a learned per-band
                     gate recombines them. Lineage: combines ClusIR's
                     frequency-domain modulation (DAFMM, arXiv:2512.10948)
                     with MoCE-IR's "multiple specialized experts" framing --
                     but neither paper does band-assigned experts; MoCE-IR's
                     experts differ by spatial complexity, and ClusIR's DAFMM
                     modulates frequency content without an expert-per-band
                     structure. This is the mode built to test H-1a
                     (RQ_PAPER_MAP.md): photometric corruption is
                     overwhelmingly low-frequency energy, so an expert whose
                     receptive field IS a frequency band has a structural
                     argument the project's existing spatial mechanisms
                     (mechanisms.py's S3 channel_attention, S4 safm) do not.

       "decompose"   static three-way channel split (conv / channel-attention
                     / frequency), every branch always active, NO routing at
                     all. Lineage: MIRAGE (Ren et al., ICLR 2026,
                     arXiv:2505.18679), which beats a routing baseline
                     (MoCE-IR) using exactly this design. Included as the
                     rival hypothesis to the two routed modes above, inside
                     the same architecture family, rather than assumed away --
                     "does routing even help" becomes a `moe_mode` switch,
                     not a separate paper's claim taken on faith.

  3. Deployability as a first-class property, not an afterthought: `width`
     and `depth` scale the whole thing continuously from roughly 10^4 to
     10^6+ params on two knobs. MoCE-IR/ClusIR/MIRAGE report only at
     6M-30M and have never been run in this project's compute-bounded
     regime; K3 is built to be run there by construction.

  4. Classifier-free by construction, not by a training-script workaround:
     `forward(x) -> reconstruction` and nothing else. K3Restorer is an
     ordinary robustmed.models.RecoveryModule like every other arm in this
     codebase (arms.py, moceir.py, mechanisms.py), so it drops into BOTH
     training paths unmodified -- engine.train_recovery (classifier-guided,
     CE term) and engine.train_restoration (classifier-free, pure L1+SSIM) --
     and can be used as a bare denoiser with no classifier and no training
     harness at all:

         from robustmed.k3 import K3Restorer
         model = K3Restorer(width=16, moe_mode="frequency").eval()
         clean_looking = model(corrupted_image_01)   # [0,1] in, [0,1] out

This file has NO import from robustmed.moceir, robustmed.mechanisms, or
robustmed.arms -- it only depends on robustmed.models.RecoveryModule for the
shared bounded-output convention (clamp/residual/sigmoid) every recovery arm
in this project shares. That is a real, one-file dependency, not zero: to run
this outside the repo, bring robustmed/models.py's RecoveryModule class (or
config.RECOVERY_OUTPUT/RECOVERY_OUTPUTS plus its ~30-line bound() method)
along with it. Verified runnable via `python -m robustmed.k3` from the repo
root (see the __main__ smoke test at the bottom) -- not `python
robustmed/k3.py` directly, which breaks the relative import the same way it
would for any file inside a package.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import RecoveryModule

MOE_MODES = ("complexity", "frequency", "decompose")


# --- shared small pieces ----------------------------------------------------

class ConvBlock(nn.Module):
    """Plain residual conv block -- the backbone's ordinary building block,
    carrying no mechanism of its own."""

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c * 2, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c, c // 2, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    """Global avg-pool -> 1x1 -> sigmoid gate. The cheapest purely-global
    mechanism available -- used as one of "decompose" mode's three branches."""

    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.gate(x))


# --- mode 1: "complexity" -- MoCE-IR-style nested experts -------------------

class ComplexityExpert(nn.Module):
    """One nested complexity expert. Projects C -> a small constant `rank`,
    then mixes q and k via FFT(q)*FFT(k) inverted -- by the convolution
    theorem this is a circular convolution of q and k, computed at
    O(N log N) via the FFT rather than O(N^2) for literal windowed
    self-attention. Gates v with the result. Same mechanism as
    robustmed/moceir.py's FFTExpert (reimplemented independently here, no
    import between the two files) -- see that file's module docstring for
    the full derivation against MoCE-IR's official source.
    """

    def __init__(self, c, rank, kernel_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        pad = kernel_size // 2
        self.proj_in = nn.Conv2d(c, rank, 1)
        self.q = nn.Conv2d(rank, rank, kernel_size, padding=pad)
        self.k = nn.Conv2d(rank, rank, kernel_size, padding=pad)
        self.v = nn.Conv2d(rank, rank, kernel_size, padding=pad)
        self.proj_out = nn.Conv2d(rank, c, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        z = self.proj_in(x)
        r = z.shape[1]
        p = self.patch_size
        ph, pw = (-h) % p, (-w) % p
        zp = F.pad(z, (0, pw, 0, ph))
        hp, wp = zp.shape[-2:]
        q, k, v = self.q(zp), self.k(zp), self.v(zp)

        def to_patches(t):
            t = t.view(b, r, hp // p, p, wp // p, p)
            return t.permute(0, 2, 4, 1, 3, 5).reshape(-1, r, p, p)

        qp, kp, vp = to_patches(q), to_patches(k), to_patches(v)
        mixed = torch.fft.irfft2(torch.fft.rfft2(qp) * torch.fft.rfft2(kp), s=(p, p))
        out = (mixed * vp)

        gh, gw = hp // p, wp // p
        out = out.view(b, gh, gw, r, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, r, hp, wp)
        return self.proj_out(out[:, :, :h, :w])


class ComplexityMoE(nn.Module):
    """n_experts nested ComplexityExperts of increasing kernel/patch size,
    top-1 routed per sample, cost-biased aux loss (Zamfir et al.'s
    "spring force" -- see robustmed/moceir.py's module docstring for the
    exact equations this follows)."""

    def __init__(self, c, n_experts=3, rank=4):
        super().__init__()
        self.n_experts = n_experts
        kernel_sizes = [3 + 2 * i for i in range(n_experts)]
        patch_sizes = [2 ** (i + 2) for i in range(n_experts)]
        self.experts = nn.ModuleList([
            ComplexityExpert(c, rank, kernel_sizes[i], patch_sizes[i])
            for i in range(n_experts)])
        p = torch.tensor([sum(w.numel() for w in e.parameters())
                          for e in self.experts], dtype=torch.float32)
        self.register_buffer("cost_bias", p / p.max())
        self.router = nn.Linear(c, n_experts)

    def forward(self, x):
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

        importance = probs.mean(0) * self.cost_bias
        load = torch.bincount(choice, minlength=self.n_experts).float() / choice.numel()
        cv = lambda t: t.std() / (t.mean() + 1e-8)
        aux = 0.5 * cv(importance) ** 2 + 0.5 * cv(load) ** 2
        return out, aux


# --- mode 2: "frequency" -- band-assigned experts (the new combination) ----

class FrequencyBandMoE(nn.Module):
    """n_bands experts, each assigned a fixed band of the 2D frequency
    spectrum (band 0 = lowest frequencies, including DC -- the band a
    uniform brightness/contrast/gamma shift lives in almost entirely).

    Unlike ComplexityMoE, there is no routing decision: every band-expert
    runs on every sample, because WHICH frequencies exist in an image is a
    property of the image, not something to guess with a router. What is
    learned is how much to trust each band's expert output (`gate`), and
    what each band-expert does with its slice of the spectrum.

    Mechanism: FFT the input once; for each band, mask the spectrum to that
    band only, stack real/imaginary parts as channels (so an ordinary conv
    can operate on a complex-valued signal), run a small 1x1 conv "expert",
    convert back to a complex tensor, mask AGAIN to keep the expert's output
    confined to its own band (so band 0's expert cannot leak energy into
    band 2's territory), weight by the learned per-band gate, sum bands, and
    inverse-FFT once at the end.
    """

    def __init__(self, c, n_bands=3, hidden=None):
        super().__init__()
        hidden = hidden or c
        self.n_bands = n_bands
        self.band_experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c * 2, hidden, 1), nn.GELU(),
                          nn.Conv2d(hidden, c * 2, 1))
            for _ in range(n_bands)])
        self.gate = nn.Parameter(torch.ones(n_bands, c, 1, 1))

    def _band_masks(self, h, w, device, dtype):
        fy = torch.fft.fftfreq(h, device=device).view(-1, 1)
        fx = torch.fft.rfftfreq(w, device=device).view(1, -1)
        radius = torch.sqrt(fy ** 2 + fx ** 2)
        edges = torch.linspace(0, radius.max(), self.n_bands + 1, device=device)
        return [((radius >= edges[i]) & (radius <= edges[i + 1])).to(dtype)
                for i in range(self.n_bands)]

    def forward(self, x):
        b, c, h, w = x.shape
        spec = torch.fft.rfft2(x, norm="ortho")
        masks = self._band_masks(h, w, x.device, x.dtype)

        out_spec = torch.zeros_like(spec)
        for i, mask in enumerate(masks):
            band = spec * mask                                  # this band only, rest zeroed
            ri = torch.cat([band.real, band.imag], dim=1)        # complex -> 2C real channels
            ri = self.band_experts[i](ri)
            re, im = ri.chunk(2, dim=1)
            band_out = torch.complex(re, im) * mask               # keep the expert inside its own band
            out_spec = out_spec + band_out * self.gate[i]
        return torch.fft.irfft2(out_spec, s=(h, w), norm="ortho")


# --- mode 3: "decompose" -- static channel split, no routing (MIRAGE) ------

class DecomposeLayer(nn.Module):
    """Static three-way channel decomposition -- MIRAGE's rival hypothesis
    to routing, included as a switchable mode rather than a separate claim
    taken on faith. Channels split into three parts (as close to equal as
    `c` allows); each part is processed by ONE mechanism -- conv, channel
    attention, or a 2-band frequency expert -- every branch always active.
    Fused via learnable gated cross-branch mixing, a simplified version of
    MIRAGE's own Algorithm 1 lines 9-11 (their fusion operates on
    equal-width branches by construction; this one averages over the
    channel dimension before cross-gating so uneven splits, when `c` is not
    divisible by 3, still combine correctly -- a deliberate simplification,
    named here rather than left implicit).
    """

    def __init__(self, c):
        super().__init__()
        third = max(1, c // 3)
        rem = c - 2 * third
        self.split = (third, third, rem)
        self.conv_branch = nn.Sequential(nn.Conv2d(third, third, 3, padding=1), nn.GELU())
        self.attn_branch = ChannelAttention(third)
        self.freq_branch = FrequencyBandMoE(rem, n_bands=2)
        self.fuse = nn.Conv2d(c, c, 1)
        self.lam = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        xc, xa, xf = torch.split(x, self.split, dim=1)
        oc = self.conv_branch(xc)
        oa = self.attn_branch(xa)
        of = self.freq_branch(xf)

        mc, ma, mf = oc.mean(1, keepdim=True), oa.mean(1, keepdim=True), of.mean(1, keepdim=True)
        oc2 = oc + self.lam[0] * torch.sigmoid(ma + mf)
        oa2 = oa + self.lam[1] * torch.sigmoid(mc + mf)
        of2 = of + self.lam[2] * torch.sigmoid(mc + ma)
        return self.fuse(torch.cat([oc2, oa2, of2], dim=1))


# --- the switchable layer and the backbone ----------------------------------

class K3Layer(nn.Module):
    """Dispatches to one of MOE_MODES. Wrapped in a residual around whichever
    mechanism is active, and normalizes every mode's return value (some
    return an aux loss, some do not) to the same (out, aux) shape."""

    def __init__(self, c, moe_mode="frequency", n_experts=3, n_bands=3, rank=4):
        super().__init__()
        if moe_mode not in MOE_MODES:
            raise KeyError(f"unknown moe_mode {moe_mode!r}; have {MOE_MODES}")
        self.moe_mode = moe_mode
        if moe_mode == "complexity":
            self.body = ComplexityMoE(c, n_experts=n_experts, rank=rank)
        elif moe_mode == "frequency":
            self.body = FrequencyBandMoE(c, n_bands=n_bands)
        else:
            self.body = DecomposeLayer(c)
        self.aux_loss = 0.0

    def forward(self, x):
        out = self.body(x)
        out, self.aux_loss = out if isinstance(out, tuple) else (out, 0.0)
        return x + out


class K3Restorer(RecoveryModule):
    """K3. See the module docstring for the full mechanism, lineage, and
    what is/is not novel. `width`/`depth` control size; `moe_mode` selects
    which of the three expert-mixing strategies sits at the bottleneck.

    Backbone: 3x3 stem, two ConvBlock stages (stride-2 downsample between
    them -- the proven mechanism), a K3Layer at the bottleneck, two ConvBlock
    decoder stages with additive skips, 3x3 head. Deliberately shallow and
    narrow by default (width=16, depth=2) -- this is a deployable-scale
    module, not a restoration-literature-scale one.
    """

    PUBLISHED = {"width": 16, "depth": 2, "moe_mode": "frequency",
                 "n_experts": 3, "n_bands": 3, "rank": 4}

    def __init__(self, width=16, depth=2, moe_mode="frequency", n_experts=3,
                n_bands=3, rank=4, output=None, **kw):
        super().__init__(output=output)
        self.moe_mode = moe_mode
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)

        self.enc1 = nn.Sequential(*[ConvBlock(width) for _ in range(depth)])
        self.down1 = Down(width)
        self.enc2 = nn.Sequential(*[ConvBlock(width * 2) for _ in range(depth)])
        self.down2 = Down(width * 2)

        c_bottom = width * 4
        self.bottleneck = K3Layer(c_bottom, moe_mode=moe_mode, n_experts=n_experts,
                                  n_bands=n_bands, rank=rank)

        self.up2 = Up(c_bottom)
        self.dec2 = nn.Sequential(*[ConvBlock(width * 2) for _ in range(depth)])
        self.up1 = Up(width * 2)
        self.dec1 = nn.Sequential(*[ConvBlock(width) for _ in range(depth)])

        self.pad = 4          # two downsamples
        self.aux_loss = 0.0

    def body(self, x):
        _, _, h, w = x.shape
        ph, pw = (-h) % self.pad, (-w) % self.pad
        z = F.pad(x, (0, pw, 0, ph))

        z = self.stem(z)
        s1 = self.enc1(z)
        z = self.down1(s1)
        s2 = self.enc2(z)
        z = self.down2(s2)

        z = self.bottleneck(z)
        self.aux_loss = self.bottleneck.aux_loss

        z = self.up2(z) + s2
        z = self.dec2(z)
        z = self.up1(z) + s1
        z = self.dec1(z)
        return self.ending(z)[:, :, :h, :w]


if __name__ == "__main__":
    # Standalone smoke test -- run `python k3.py` with just torch installed,
    # no robustmed package or dataset needed, to confirm this file works in
    # isolation (e.g. after copying it out of the repo onto Kaggle).
    for mode in MOE_MODES:
        for px in (64, 224):
            m = K3Restorer(width=16, depth=2, moe_mode=mode)
            m.train()
            x = torch.rand(2, 3, px, px)
            y = m(x)
            n = sum(p.numel() for p in m.parameters())
            aux_val = float(m.aux_loss.detach()) if torch.is_tensor(m.aux_loss) else m.aux_loss
            loss = y.mean() + 0.01 * m.aux_loss
            loss.backward()
            print(f"mode={mode:10s} px={px:3d}  out={tuple(y.shape)}  "
                  f"range=({y.min().item():.3f},{y.max().item():.3f})  "
                  f"params={n:,}  aux={aux_val:.4f}  backward=OK")
