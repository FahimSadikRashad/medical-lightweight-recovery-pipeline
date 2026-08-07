"""Stage 2 comparison arms, at their PUBLISHED configurations.

An earlier version of this file shrank every architecture to 2-5 blocks at
width<=32. That produced a table in which the plain ConvAE beat NAFNet, SPAN and
SAFMN -- but the result was an artifact of the shrinking, not a finding. NAFNet
*is* a U-Net and SAFM *is* multi-level pooling; removing the multi-scale
structure from exactly the architectures whose defining feature is multi-scale,
while leaving ConvAE's two stride-2 convs intact, handed the baseline a
receptive-field advantage its competitors had been stripped of.

So each class below defaults to the configuration its paper reports, in PUBLISHED.
`width=None` builds that. Passing an explicit width scales the model down for the
compute-bounded sweep, and the two must be reported as different things: the
published config says what the motif achieves, the scaled one says what survives
at a few thousand parameters.

One adaptation applies to all of them and cannot be avoided: the published SPAN
and SAFMN end in PixelShuffle for 4x super-resolution. Our task is
same-resolution restoration in front of a frozen classifier, so the upsampling
head is replaced by a plain conv. Everything before it is unchanged. State this
in the paper -- it is a real deviation, just an unavoidable one.

Where a detail could not be verified it is flagged inline rather than presented
as faithful. Check those against the papers before the camera-ready.

PARAMETER-COUNT CHECK, resolved against the papers and official configs:

    arm     built    reported                            verdict
    DnCNN    558k    ~556k (DnCNN-S, 17 layers, 64ch)    matches
    SAFMN    225k    ~240k (dim 36, 8 blocks)            matches
    NAFNet    29M    see below                           EXACT
    SPAN     398k    498k at x4 SR                       explained

NAFNet: the 29M here is the SIDD config (width 32, enc [2,2,4,8], middle 12,
dec [2,2,2,2]). The widely quoted 17.11M is a DIFFERENT config -- GoPro
width32, enc [1,1,1,28], middle 1, dec [1,1,1,1]. This implementation returns
17,111,907 for the GoPro config and 67,888,835 for GoPro width64, both exact
against the paper, so the block is right and only the reference number was
wrong. SIDD is the denoising config and therefore the one to use here.

SPAN: reported is 498K, not the 150K quoted earlier. The 100K difference from
our 398K is the x4 PixelShuffle head we necessarily drop plus SPAN's wider
feature-concatenation conv. The body matches: 6 blocks x 3 convs x 3x3 x 48ch.

The only genuine implementation error found was in SPAB's attention -- see the
note there. It is fixed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import RecoveryModule


class DnCNN(RecoveryModule):
    """Zhang et al., TIP 2017. DnCNN-S: 17 layers, 64 channels (~556k params).

    Conv-ReLU, then 15 x Conv-BN-ReLU, then Conv. Predicts a residual, which the
    shared output parameterization already provides.

    Present as the control: one fixed learned filter bank applied identically to
    every input, designed for additive Gaussian noise. If that suffices, no
    conditional architecture is needed and K3 has no reason to exist.
    """

    PUBLISHED = {"width": 64, "depth": 17}

    def __init__(self, width=64, depth=17, output=None, **kw):
        super().__init__(output=output)
        self.width = width
        layers = [nn.Conv2d(3, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1, bias=False),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def body(self, x):
        return self.net(x)


# --- NAFNet ----------------------------------------------------------------

class SimpleGate(nn.Module):
    """Split channels in half and multiply. No parameters, no activation."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """Chen et al., ECCV 2022, Figure 4. dw_expand=2, ffn_expand=2.

    Costs ~7c^2 per block, which reproduces the paper's totals exactly at both
    published GoPro configs (17,111,907 at width32 and 67,888,835 at width64).
    """

    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        d = c * dw_expand
        self.norm1 = nn.GroupNorm(1, c)          # LayerNorm2d
        self.conv1 = nn.Conv2d(c, d, 1)
        self.conv2 = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(d // 2, d // 2, 1))
        self.conv3 = nn.Conv2d(d // 2, c, 1)

        f = c * ffn_expand
        self.norm2 = nn.GroupNorm(1, c)
        self.conv4 = nn.Conv2d(c, f, 1)
        self.conv5 = nn.Conv2d(f // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.sg(self.conv2(self.conv1(self.norm1(x))))
        y = self.conv3(y * self.sca(y))
        x = x + y * self.beta
        y = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + y * self.gamma


class NAFNet(RecoveryModule):
    """NAFNet, full U-Net. Published: width 32, enc [2,2,4,8], middle 12,
    dec [2,2,2,2] (~17M params).

    The multi-scale encoder-decoder is restored here. It is the reason the
    architecture works and the thing the previous single-scale port removed.

    On a 64x64 input the four downsampling stages reach 4x4 at the bottleneck,
    so the published depth is usable without modification.
    """

    PUBLISHED = {"width": 32, "enc_blks": (2, 2, 4, 8), "middle_blks": 12,
                 "dec_blks": (2, 2, 2, 2)}

    def __init__(self, width=32, enc_blks=(2, 2, 4, 8), middle_blks=12,
                 dec_blks=(2, 2, 2, 2), output=None, **kw):
        super().__init__(output=output)
        self.width = width
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)

        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()

        c = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, 2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blks)])
        for n in dec_blks:
            # pixelshuffle upsample, as in the paper
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.pad = 2 ** len(enc_blks)

    def body(self, x):
        _, _, h, w = x.shape
        ph, pw = (-h) % self.pad, (-w) % self.pad
        z = F.pad(x, (0, pw, 0, ph))

        z = self.intro(z)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            z = enc(z)
            skips.append(z)
            z = down(z)
        z = self.middle(z)
        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            z = up(z) + skip
            z = dec(z)
        return self.ending(z)[:, :, :h, :w]


# --- SPAN ------------------------------------------------------------------

class SPAB(nn.Module):
    """Wan et al., NTIRE 2024 -- Swift Parameter-free Attention Block.

    Three 3x3 convs; the attention map is derived from the pre-activation
    features by a symmetric activation, so attention costs zero parameters.

    Per arXiv:2311.12770 (v3), Section 3:
      H  = three 3x3 convs, all c-channeled
      U  = O_{i-1} (+) H          residual added BEFORE attention
      A  = sigma_a(H),  sigma_a(x) = Sigmoid(x) - 0.5
      O  = U * A

    sigma_a is symmetric about the origin, which is the property the paper
    requires: x * sigma_a(x) > 0, so gradients are amplified in information-rich
    regions and suppressed elsewhere. A plain sigmoid is NOT symmetric about the
    origin and gives away exactly that property -- which is what this originally
    used.
    """

    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.c3 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = F.silu(self.c1(x))
        h = F.silu(self.c2(h))
        h = self.c3(h)
        u = x + h                                  # pre-attention feature map
        return u * (torch.sigmoid(h) - 0.5)        # symmetric about the origin


class SPAN(RecoveryModule):
    """SPAN. Published: 48 feature channels, 6 SPAB blocks.

    The efficiency reference point for this literature -- NTIRE 2025 and 2026
    both define their constraint as beating SPAN on runtime, parameters and
    FLOPs, which is what makes our cost numbers commensurable with that work.

    Adapted: the published model ends in PixelShuffle for 4x SR; replaced by a
    plain conv for same-resolution restoration.
    """

    PUBLISHED = {"width": 48, "blocks": 6}

    def __init__(self, width=48, blocks=6, output=None, **kw):
        super().__init__(output=output)
        self.width = width
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.Sequential(*[SPAB(width) for _ in range(blocks)])
        self.cat = nn.Conv2d(width, width, 3, padding=1)
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    def body(self, x):
        z = self.stem(x)
        z = self.cat(self.blocks(z)) + z
        return self.head(z)


# --- SAFMN -----------------------------------------------------------------

class SAFM(nn.Module):
    """Sun et al., ICCV 2023 -- Spatially-Adaptive Feature Modulation.

    Channels split into `levels` groups; group i is pooled to 1/2^i, passed
    through a 3x3 depthwise conv, and upsampled back. The concatenation is
    projected and used to MODULATE the input by multiplication.

    Published uses 4 levels. This is the closest published motif to K3's gate,
    so it is the prior-art check: if SAFM alone fixes the photometric category,
    the gated module is not a contribution and we need to know before building it.
    """

    def __init__(self, c, levels=4):
        super().__init__()
        self.levels = levels
        self.split = c // levels
        self.convs = nn.ModuleList([
            nn.Conv2d(self.split, self.split, 3, padding=1, groups=self.split)
            for _ in range(levels)])
        self.out = nn.Conv2d(c, c, 1)
        self.act = nn.GELU()

    def forward(self, x):
        parts = torch.split(x, self.split, dim=1)
        h, w = x.shape[-2:]
        done = []
        for i, (p, conv) in enumerate(zip(parts, self.convs)):
            if i == 0:
                done.append(conv(p))
            else:
                s = 2 ** i
                q = F.adaptive_max_pool2d(p, (max(1, h // s), max(1, w // s)))
                q = conv(q)
                done.append(F.interpolate(q, size=(h, w), mode="nearest"))
        return x * self.act(self.out(torch.cat(done, dim=1)))


class CCM(nn.Module):
    """SAFMN's convolutional channel mixer: 3x3 expand, GELU, 1x1 project."""

    def __init__(self, c, ffn_scale=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, int(c * ffn_scale), 3, padding=1), nn.GELU(),
            nn.Conv2d(int(c * ffn_scale), c, 1))

    def forward(self, x):
        return self.net(x)


class SAFMBlock(nn.Module):
    def __init__(self, c, levels=4, ffn_scale=2):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(1, c), nn.GroupNorm(1, c)
        self.safm, self.ccm = SAFM(c, levels), CCM(c, ffn_scale)

    def forward(self, x):
        x = x + self.safm(self.n1(x))
        return x + self.ccm(self.n2(x))


class SAFMNet(RecoveryModule):
    """SAFMN. Published: dim 36, 8 blocks, 4 SAFM levels, ffn_scale 2.

    Adapted: PixelShuffle SR head replaced by a plain conv.

    dim must be divisible by `levels`; 36/4 = 9, which is why the published
    width works unmodified.
    """

    PUBLISHED = {"width": 36, "blocks": 8, "levels": 4}

    def __init__(self, width=36, blocks=8, levels=4, output=None, **kw):
        super().__init__(output=output)
        # SAFM splits channels evenly, so a scaled-down width has to stay
        # divisible by the level count or whole groups are silently dropped.
        while levels > 1 and width % levels:
            levels -= 1
        self.width, self.levels = width, levels
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.Sequential(*[SAFMBlock(width, levels)
                                      for _ in range(blocks)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    def body(self, x):
        return self.head(self.blocks(self.stem(x)))
