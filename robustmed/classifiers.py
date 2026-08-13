"""MedViT: a hybrid CNN-Transformer classifier, ported as a second frozen backbone.

Manzari et al., Computers in Biology and Medicine 2023, arXiv:2302.09462 --
"MedViT: A Robust Vision Transformer for Generalized Medical Image
Classification."

This is the classifier-side counterpart to arms.py: an independently-published
architecture ported in as a new implementation, not invented for this project.
Its role is the FROZEN baseline that the recovery module sits in front of (see
models.Classifier) -- run as a SECOND backbone alongside MobileNetV2, not a
replacement, so classifier-coupling (RQ-5 / H-1d / H-5a in
docs/RQ_PAPER_MAP.md) finally has something architecturally different to test
against instead of two CNNs of different sizes.

Architecture, verified against the official implementation
(github.com/Omid-Nejati/MedViT/blob/main/MedViT.py), MedViT_small config:

    stem_chs   = (64, 32, 64)     three convs, stride (2, 1, 2)
    depths     = (3, 4, 10, 3)    blocks per stage
    strides    = (1, 2, 2, 2)     downsample at stage entry -- stage 0 does not
    sr_ratios  = (8, 4, 2, 1)     E-MHSA spatial reduction, per stage
    head_dim   = 32               both MHCA and E-MHSA
    stage channels: [96]*3, [192]*3+[256], ([384]*4+[512])*2, [768]*2+[1024]
    stage block types: LLL, LLLG, LLLLG LLLLG, LLG  (L=LFP/"ECB", G=GFP/"LTB")

Each stage is (LFP block)*N followed by one GFP block: locality first,
globality last per stage, at the resolution where a full attention map is
cheapest. LFP = "ECB" (efficient convolution block) in the paper; GFP = "LTB"
(local transformer block). Renamed here to match this project's recovery-side
vocabulary (Local/Global Feature Perception), not the paper's abbreviations.

    LFP block:  patch-embed (downsamples iff this block opens the stage)
                -> MHCA, a grouped 3x3 conv acting as a cheap local-attention
                   surrogate (residual)
                -> inverted-residual FFN: 1x1 expand x3, depthwise 3x3,
                   squeeze-excite, 1x1 project (residual)

    GFP block:  patch-embed
                -> E-MHSA: standard scaled dot-product attention with K/V
                   spatially pooled by sr_ratio before the dot product -- the
                   trick that keeps a global attention map affordable at the
                   early, high-resolution stages (residual)
                -> MHCA (residual)
                -> inverted-residual FFN (residual)

Deviations from the official implementation, named rather than taken silently:

  - No ImageNet-pretrained weights are ported. MobileNet_V2_Weights.IMAGENET1K_V1
    is a torchvision one-liner; MedViT ships a project-specific .pth this
    codebase has no fetch path for. `pretrained=True` is therefore a no-op here
    (prints a note once) -- MedViT trains from scratch through Baseline 1/2
    like everything else in this pipeline. Its early-epoch numbers are NOT
    comparable to MobileNetV2's until that gap is closed; say so in the paper.
  - The paper splits GFP's hidden width between the E-MHSA and MHCA branches by
    `mix_block_ratio` (0.75 in the official 'small' config). This port runs
    both branches over the full width instead -- an even split, not the
    paper's 3:1 -- because the official split point is defined over an
    internal dimension this port does not carry. Check against the source
    before citing a matched parameter count the way arms.py does for the
    recovery arms; this file has not had that pass yet.
  - Only the 'small' depths=(3,4,10,3) config is exercised. 'base'/'large'
    change the repeat count inside stage 2 (`depths[2] // 5` repeats of the
    5-block LLLLG pattern) and the stage-plan logic below supports any depths
    divisible the same way, but nothing downstream selects them.
  - Requires IMAGE_SIZE divisible by 32 (stem /4, three stage downsamples /8).
    Both resolutions this project uses -- 64 and 224 -- satisfy that; no
    padding logic like NAFNet's is included because it has not been needed.

PARAMETER-COUNT CHECK. This port: 74,256,458 params (both 64px and 224px --
global pooling makes the head resolution-independent). MobileNetV2 in this
codebase, per models.backbone, is ~2.2M. That is a >30x size gap, not just an
architecture difference, and it is a genuine confound for the classifier-
coupling question (RQ-5 / H-1d / H-5a) this file exists to unblock: a
difference observed between the two backbones cannot be attributed to
"different mechanism" versus "different capacity" without a matched-scale
control. Flagged rather than absorbed -- narrow depths/stem_chs to close the
gap before that comparison is reported, the same discipline arms.py applies
via PUBLISHED vs scaled configs for the recovery arms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- shared primitives -------------------------------------------------------

class SqueezeExcite(nn.Module):
    """Channel gate: global-avg-pool -> 1x1 -> ReLU -> 1x1 -> sigmoid -> scale."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, stride=1, groups=1, act=True):
        layers = [nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(out_ch)]
        if act:
            layers.append(nn.Hardswish(inplace=True))
        super().__init__(*layers)


class PatchEmbed(nn.Module):
    """Stage/block-entry adapter: identity when shape already matches,
    otherwise a plain conv that handles both the channel change and any
    downsampling stride in one step."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.proj = (nn.Identity() if stride == 1 and in_ch == out_ch
                    else ConvBNAct(in_ch, out_ch, k=3, stride=stride, act=False))

    def forward(self, x):
        return self.proj(x)


class MHCA(nn.Module):
    """Multi-head convolutional attention: one group per head_dim channels,
    3x3 grouped conv, then a 1x1 projection. No residual here -- the block
    that owns this adds it, matching the shared-output convention in
    models.RecoveryModule (one place decides the residual, not every module)."""

    def __init__(self, channels, head_dim=32):
        super().__init__()
        groups = max(1, channels // head_dim)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return self.proj(self.act(self.norm(self.conv(x))))


class EMHSA(nn.Module):
    """Efficient multi-head self-attention. K/V are spatially average-pooled
    by sr_ratio before the dot product; queries stay at full resolution. That
    asymmetry is what keeps the attention map affordable at early, large
    feature maps -- full self-attention at stage 0's resolution would dominate
    the whole model's compute."""

    def __init__(self, channels, head_dim=32, sr_ratio=1):
        super().__init__()
        self.heads = max(1, channels // head_dim)
        self.head_dim = channels // self.heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels * 2)
        self.proj = nn.Linear(channels, channels)
        if sr_ratio > 1:
            self.pool = nn.AvgPool2d(sr_ratio, sr_ratio)
            self.pool_norm = nn.BatchNorm2d(channels)

    def _split_heads(self, t, b):
        return t.view(b, -1, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        b, c, h, w = x.shape
        q_tok = x.flatten(2).transpose(1, 2)                      # B,N,C
        kv_src = self.pool_norm(self.pool(x)) if self.sr_ratio > 1 else x
        kv_tok = kv_src.flatten(2).transpose(1, 2)                 # B,N',C

        q = self._split_heads(self.q(q_tok), b)
        k, v = (self._split_heads(t, b) for t in self.kv(kv_tok).chunk(2, dim=-1))

        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, h * w, c)
        return self.proj(out).transpose(1, 2).reshape(b, c, h, w)


class LocalityFeedForward(nn.Module):
    """Inverted-residual FFN: 1x1 expand, depthwise 3x3, squeeze-excite,
    1x1 project. expand_ratio=3 matches the official 'small' config."""

    def __init__(self, channels, expand_ratio=3, se_reduction=4):
        super().__init__()
        hidden = channels * expand_ratio
        self.expand = ConvBNAct(channels, hidden, k=1)
        self.dw = ConvBNAct(hidden, hidden, k=3, groups=hidden)
        self.se = SqueezeExcite(hidden, se_reduction)
        self.project = nn.Sequential(nn.Conv2d(hidden, channels, 1, bias=False),
                                     nn.BatchNorm2d(channels))

    def forward(self, x):
        h = self.dw(self.expand(x))
        h = self.se(h)
        return self.project(h)


# --- LFP / GFP blocks --------------------------------------------------------

class LFPBlock(nn.Module):
    """Local Feature Perception -- 'ECB' in the paper."""

    def __init__(self, in_ch, out_ch, stride, head_dim=32):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, out_ch, stride)
        self.mhca = MHCA(out_ch, head_dim)
        self.ffn = LocalityFeedForward(out_ch)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.mhca(x)
        x = x + self.ffn(x)
        return x


class GFPBlock(nn.Module):
    """Global Feature Perception -- 'LTB' in the paper. See the module
    docstring for the mix_block_ratio deviation (even split here, 3:1 in the
    paper)."""

    def __init__(self, in_ch, out_ch, stride, sr_ratio, head_dim=32):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, out_ch, stride)
        self.attn = EMHSA(out_ch, head_dim, sr_ratio)
        self.mhca = MHCA(out_ch, head_dim)
        self.ffn = LocalityFeedForward(out_ch)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attn(x)
        x = x + self.mhca(x)
        x = x + self.ffn(x)
        return x


def _stage_plan(depths):
    """Per-stage (channels, block-type) lists, verified against the official
    MedViT_small constructor:

        stage_out_channels = [[96]*d0,
                               [192]*(d1-1) + [256],
                               ([384,384,384,384,512]) * (d2 // 5),
                               [768]*(d3-1) + [1024]]
        stage_block_types  = [[ECB]*d0,
                               [ECB]*(d1-1) + [LTB],
                               [ECB,ECB,ECB,ECB,LTB] * (d2 // 5),
                               [ECB]*(d3-1) + [LTB]]

    'L' stands in for ECB, 'G' for LTB -- see the module docstring for why.
    """
    d0, d1, d2, d3 = depths
    reps = d2 // 5
    channels = [
        [96] * d0,
        [192] * (d1 - 1) + [256],
        ([384, 384, 384, 384, 512] * reps)[:d2],
        [768] * (d3 - 1) + [1024],
    ]
    types = [
        ["L"] * d0,
        ["L"] * (d1 - 1) + ["G"],
        (["L", "L", "L", "L", "G"] * reps)[:d2],
        ["L"] * (d3 - 1) + ["G"],
    ]
    return channels, types


class MedViT(nn.Module):
    """Manzari et al., 2023. See the module docstring for the verified config
    and the deviations from it. Call with no arguments (or **PUBLISHED) for
    MedViT_small, the only variant exercised in this pipeline.

    `pretrained` is accepted for interface parity with models.backbone() but
    is always a no-op here -- see the module docstring's first deviation.
    """

    PUBLISHED = {"stem_chs": (64, 32, 64), "depths": (3, 4, 10, 3),
                 "strides": (1, 2, 2, 2), "sr_ratios": (8, 4, 2, 1),
                 "head_dim": 32}

    def __init__(self, n_classes, pretrained=False, stem_chs=(64, 32, 64),
                depths=(3, 4, 10, 3), strides=(1, 2, 2, 2),
                sr_ratios=(8, 4, 2, 1), head_dim=32, **kw):
        super().__init__()
        c0, c1, c2 = stem_chs
        self.stem = nn.Sequential(
            ConvBNAct(3, c0, k=3, stride=2),
            ConvBNAct(c0, c1, k=3, stride=1),
            ConvBNAct(c1, c2, k=3, stride=2),
        )

        channels, types = _stage_plan(depths)
        self.stages = nn.ModuleList()
        in_ch = c2
        for chs, tys, stage_stride, sr in zip(channels, types, strides, sr_ratios):
            blocks = []
            for i, (ch, ty) in enumerate(zip(chs, tys)):
                blk_stride = stage_stride if i == 0 else 1
                blocks.append(LFPBlock(in_ch, ch, blk_stride, head_dim) if ty == "L"
                             else GFPBlock(in_ch, ch, blk_stride, sr, head_dim))
                in_ch = ch
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.BatchNorm2d(in_ch)
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        return self.head(x.mean((2, 3)))
