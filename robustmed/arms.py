"""Stage 2 comparison arms: efficient-restoration design motifs at our scale.

READ THIS BEFORE CITING ANY OF IT. Every architecture here was published at
>=100k parameters, most of them for super-resolution. What is reproduced is each
paper's *design motif* -- the mechanism the paper argues for -- rebuilt at a few
thousand parameters and stripped of the upsampling head, because our task is
same-resolution restoration in front of a frozen classifier. These are not
reimplementations, and results must never be written up as "we compare against
NAFNet". The claim is "we port efficient-restoration design motifs to the
compute-bounded regime", and each class below states exactly what was kept and
what was changed.

Why these motifs. Stage 1 refuted the assumption that drove the original arm
list: blur turned out to be the best-recovered category (+0.172 mean gain) while
photometric corruption was the binding one (-0.056, i.e. worse than leaving the
image alone). So the axis that matters is not high-frequency reconstruction but
CONDITIONAL behaviour -- whether an architecture can modulate what it does per
input, including doing nothing. That is why SAFM's feature modulation and SPAN's
input-derived attention are here, and why DnCNN, which applies one fixed learned
filter bank, is the control that should lose.

All arms subclass RecoveryModule, so they share one gradient-safe bounded output
and the comparison is not confounded by some of them carrying a global residual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import RecoveryModule


class DnCNN(RecoveryModule):
    """Zhang et al., TIP 2017. Control, not a contender.

    Motif kept: a plain stack of conv-BN-ReLU predicting a residual correction.
    Changed: depth and width shrunk from 17x64 to depth x width.

    It is here to lose, and the reason is stated in advance: DnCNN learns one
    fixed filter bank applied identically to every input. It was designed for
    additive Gaussian noise, which Stage 1 shows is already the category
    recovery handles well. If a fixed filter bank were enough, no conditional
    architecture would be needed and K3 would be pointless -- so this arm is the
    null hypothesis for the whole paper.
    """

    def __init__(self, width=16, depth=5, output=None, **kw):
        super().__init__(output=output)
        self.width = width
        layers = [nn.Conv2d(3, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Conv2d(width, width, 3, padding=1, bias=False),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def body(self, x):
        return self.net(x)


class SimpleGate(nn.Module):
    """NAFNet's activation: split channels in half and multiply. No parameters.

    The paper's central claim is that this replaces GELU/attention nonlinearity
    at lower cost, which is exactly the kind of trade that matters at our scale.
    """

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """Chen et al., ECCV 2022 -- Nonlinear Activation Free block.

    Kept: LayerNorm -> 1x1 expand -> depthwise 3x3 -> SimpleGate -> simplified
    channel attention -> 1x1 project, then an FFN with the same gate, each with
    a learned residual scale. Changed: expansion factors reduced to keep the
    block affordable at width<=32.

    Note the simplified channel attention is global average pooling, so this
    block CAN in principle represent a global intensity adjustment -- which
    makes it the most interesting non-gated arm given Stage 1's photometric
    result.
    """

    def __init__(self, c, expand=2):
        super().__init__()
        d = c * expand
        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(c, d, 1)
        self.dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.gate1 = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d // 2, d // 2, 1))
        self.conv2 = nn.Conv2d(d // 2, c, 1)

        self.norm2 = nn.GroupNorm(1, c)
        self.conv3 = nn.Conv2d(c, d, 1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(d // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.gate1(self.dw(self.conv1(self.norm1(x))))
        y = self.conv2(y * self.sca(y))
        x = x + y * self.beta
        y = self.conv4(self.gate2(self.conv3(self.norm2(x))))
        return x + y * self.gamma


class NAFNet(RecoveryModule):
    """NAFNet motif at our scale: stem, N NAFBlocks, head. No U-Net encoder.

    Changed: the published model is a multi-scale U-Net. At width<=32 the
    downsampling path costs more than it returns on 64x64 inputs, so this is the
    single-scale variant -- the block is the contribution, the U-Net is not.
    """

    def __init__(self, width=16, blocks=2, output=None, **kw):
        super().__init__(output=output)
        self.width = width
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.Sequential(*[NAFBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    def body(self, x):
        return self.head(self.blocks(self.stem(x)))


class SPAB(nn.Module):
    """Wan et al., NTIRE 2024 -- Swift Parameter-free Attention Block.

    Kept the motif that names the paper: the attention map is computed from the
    features themselves through a parameter-free symmetric activation, so
    attention costs zero parameters. Changed: channel counts scaled down.

    This is the cheapest form of input-conditional behaviour available, which is
    what Stage 1 says we should be testing.
    """

    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.c3 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        y = F.silu(self.c1(x))
        y = F.silu(self.c2(y))
        y = self.c3(y)
        # parameter-free attention: symmetric about 0, derived from y alone
        return x + y * torch.sigmoid(y)


class SPAN(RecoveryModule):
    """SPAN motif. The efficiency reference point for this literature.

    NTIRE 2025 and 2026 both define their efficiency constraint as beating SPAN
    on runtime, parameters and FLOPs, so including it is what makes our cost
    numbers commensurable with that work.

    Changed: the published SPAN ends in PixelShuffle for 4x super-resolution.
    Same-resolution restoration has no upsampling stage, so the head is a plain
    conv. The SPAB stack -- the actual contribution -- is unchanged in structure.
    """

    def __init__(self, width=16, blocks=3, output=None, **kw):
        super().__init__(output=output)
        self.width = width
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.blocks = nn.Sequential(*[SPAB(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    def body(self, x):
        return self.head(self.blocks(self.stem(x)))


class SAFM(nn.Module):
    """Sun et al., ICCV 2023 -- Spatially-Adaptive Feature Modulation.

    Channels are split into `levels` groups, each processed at a different
    spatial scale, then recombined and used to MODULATE the input by
    multiplication rather than to replace it.

    This is the closest published motif to K3's gate, which is why it matters
    here: it is the honest prior-art check. If SAFM alone fixes the photometric
    category, the gated module in Stage 3 is not a contribution and we should
    know that before building it.
    """

    def __init__(self, c, levels=2):
        super().__init__()
        self.levels = levels
        self.split = c // levels
        self.convs = nn.ModuleList([
            nn.Conv2d(self.split, self.split, 3, padding=1, groups=self.split)
            for _ in range(levels)])
        self.out = nn.Conv2d(self.split * levels, c, 1)

    def forward(self, x):
        parts = torch.split(x, self.split, dim=1)[:self.levels]
        done = []
        for i, (p, conv) in enumerate(zip(parts, self.convs)):
            if i == 0:
                done.append(conv(p))
            else:                       # coarser scale, then back up
                h, w = p.shape[-2:]
                s = max(1, 2 ** i)
                q = F.adaptive_max_pool2d(p, (max(1, h // s), max(1, w // s)))
                q = conv(q)
                done.append(F.interpolate(q, size=(h, w), mode="nearest"))
        return x * torch.sigmoid(self.out(torch.cat(done, dim=1)))


class SAFMNet(RecoveryModule):
    """SAFMN motif: stem, N x (SAFM + channel mixer), head.

    Changed: upsampling head removed as with SPAN; block count and width scaled
    to our regime.
    """

    def __init__(self, width=16, blocks=2, output=None, **kw):
        super().__init__(output=output)
        self.width = width
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        blks = []
        for _ in range(blocks):
            blks += [SAFM(width), nn.Conv2d(width, width, 1), nn.GELU()]
        self.blocks = nn.Sequential(*blks)
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    def body(self, x):
        return self.head(self.blocks(self.stem(x)))
