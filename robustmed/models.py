"""Classifier, recovery modules, and the non-learned controls."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _classifiers():
    """Imported lazily, same reason arms/mechanisms are: keeps a plain
    `import robustmed.models` from dragging in the classifier zoo."""
    from . import classifiers
    return {"medvit": classifiers.MedViT}


CLASSIFIER_NAMES = ("mobilenet_v2", "medvit")


def backbone(n_classes, pretrained=True, arch="mobilenet_v2"):
    """The frozen backbone, dispatched by name.

    Defined once. The original notebook redefined MobileNetV2 five times --
    that is how the two baselines silently drift apart. `arch` is the second
    frozen backbone this project runs alongside it -- see classifiers.py for
    why (RQ-5, classifier-agnostic recovery) and for the size-confound this
    introduces (>30x params vs MobileNetV2, not matched).
    """
    if arch == "mobilenet_v2":
        from torchvision import models
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v2(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
        return m

    zoo = _classifiers()
    if arch not in zoo:
        raise KeyError(f"unknown classifier arch {arch!r}; have {CLASSIFIER_NAMES}")
    cls = zoo[arch]
    if pretrained:
        print(f"note: no pretrained weights available for classifier arch "
              f"{arch!r} -- training from scratch (see classifiers.py)")
    return cls(n_classes, pretrained=False, **getattr(cls, "PUBLISHED", {}))


class Classifier(nn.Module):
    """Takes [0,1] tensors and normalizes internally.

    Keeping normalization here is what lets one loader feed both the AE
    (pixel space) and the classifier -- and it is shared across every
    classifier arch, not just MobileNetV2, so the [0,1]/ImageNet-normalized
    contract in data.py never has to know which backbone is frozen behind it.
    """

    def __init__(self, n_classes, pretrained=True, arch=None):
        super().__init__()
        self.arch = arch or config.CLF_ARCH
        self.net = backbone(n_classes, pretrained, arch=self.arch)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x01):
        return self.net((x01 - self.mean) / self.std)

    def freeze(self):
        """Frozen for the rest of the project, so any recovered accuracy is
        attributable to the recovery module and not to the classifier learning
        the corruptions."""
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self


class RecoveryModule(nn.Module):
    """Base class for every recovery arm: [0,1] -> [0,1].

    Subclasses implement `body(x)` and inherit one shared output
    parameterization. That sharing is the point, not a convenience: the output
    nonlinearity decides whether a tiny module trains at all, so letting each
    architecture pick its own would confound the module comparison. Most modern
    restoration architectures carry a global residual and would be immune to the
    clamp trap below, making them look more stable than a plain encoder-decoder
    for a reason that has nothing to do with their design.

    output modes (config.RECOVERY_OUTPUT):
      "clamp"     clamp(body(x)). ZERO gradient outside [0,1]. A decoder that
                  initialises negative everywhere is dead permanently -- this
                  produced 9 of 20 dead runs in the stability sweep, including
                  two whose loss froze at exactly 0.3572 + lam*4.2037 for ten
                  epochs. Kept only to reproduce that failure on demand.
      "residual"  clamp(x + body(x)). Output starts at the input, so the clamp
                  is nowhere near saturation at init.
      "sigmoid"   sigmoid(body(x)). Gradient is never exactly zero.
    """

    def __init__(self, output=None):
        super().__init__()
        self.output = output or config.RECOVERY_OUTPUT
        if self.output not in config.RECOVERY_OUTPUTS:
            raise ValueError(f"output must be one of {config.RECOVERY_OUTPUTS}")

    def body(self, x):
        raise NotImplementedError

    def bound(self, x, raw):
        if self.output == "sigmoid":
            return torch.sigmoid(raw)
        if self.output == "residual":
            return torch.clamp(x + raw, 0.0, 1.0)
        return torch.clamp(raw, 0.0, 1.0)

    def forward(self, x):
        return self.bound(x, self.body(x))

    @torch.no_grad()
    def saturation(self, x):
        """Fraction of output elements sitting on a zero-gradient boundary.

        The leading indicator of a dead run: a module at 1.0 here cannot learn,
        because clamp passes no gradient through a saturated element. Logged per
        epoch so a collapse is visible while it happens rather than afterwards.
        """
        raw = self.body(x)
        if self.output == "sigmoid":
            return float((raw.abs() > 12).float().mean())
        pre = x + raw if self.output == "residual" else raw
        return float(((pre < 0) | (pre > 1)).float().mean())


class ConvAE(RecoveryModule):
    """Naive baseline arm: plain conv encoder-decoder. width scales capacity.

    This is the module the project started with. It is a baseline in the
    comparison, not the subject of it.
    """

    def __init__(self, width=16, residual=None, output=None):
        # `residual` is the legacy flag. It used to select x+delta directly;
        # that behaviour now lives in output="residual", so map it across and
        # keep old call sites and checkpoints working.
        if output is None:
            output = "residual" if residual else config.RECOVERY_OUTPUT
        super().__init__(output=output)
        w = width
        self.width = width
        self.residual = self.output == "residual"
        self.enc = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1), nn.ReLU(inplace=True),        # 64 -> 32
            nn.Conv2d(w, w * 2, 3, 2, 1), nn.ReLU(inplace=True),    # 32 -> 16
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w * 2, w, 4, 2, 1), nn.ReLU(inplace=True),  # 16 -> 32
            nn.ConvTranspose2d(w, 3, 4, 2, 1),                             # 32 -> 64
        )

    def body(self, x):
        return self.dec(self.enc(x))


class BoxDenoiser(nn.Module):
    """Non-learned control: fixed 3x3 box filter, zero parameters.

    Without this ablation row, "the AE recovers accuracy" isn't yet a claim
    about learning -- trivial smoothing might do the same.
    """

    def __init__(self, k=3):
        super().__init__()
        self.register_buffer("kernel", torch.ones(3, 1, k, k) / (k * k))
        self.pad = k // 2

    def forward(self, x):
        return torch.clamp(F.conv2d(x, self.kernel, padding=self.pad, groups=3), 0, 1)


class UnsharpFilter(nn.Module):
    """Non-learned control: fixed unsharp mask, zero parameters.

    The counterpart to BoxDenoiser and the honest control for the
    objectives-conflict claim. Blur is the binding condition in the sweep, and
    blur wants a HIGH-pass operation -- so before crediting any learned module
    with restoring blurred images, a fixed sharpener has to be shown not to do
    the same thing. `amount` is the classic unsharp strength.
    """

    def __init__(self, k=3, amount=1.0):
        super().__init__()
        self.register_buffer("kernel", torch.ones(3, 1, k, k) / (k * k))
        self.pad, self.amount = k // 2, amount

    def forward(self, x):
        low = F.conv2d(x, self.kernel, padding=self.pad, groups=3)
        return torch.clamp(x + self.amount * (x - low), 0, 1)


# --- arm registry ----------------------------------------------------------
NON_LEARNED = {"identity": None, "box": BoxDenoiser, "unsharp": UnsharpFilter}


def _learned():
    """Imported lazily so robustmed.arms can import RecoveryModule from here."""
    from . import arms
    from . import mechanisms
    from . import moceir
    return {"convae": ConvAE, "dncnn": arms.DnCNN, "nafnet": arms.NAFNet,
            "span": arms.SPAN, "safmn": arms.SAFMNet,
            "mech": mechanisms.MechanismNet, "moceir": moceir.MoCEIR}


LEARNED_NAMES = ("convae", "dncnn", "nafnet", "span", "safmn", "mech", "moceir")
ARM_NAMES = tuple(NON_LEARNED) + LEARNED_NAMES


def build_recovery(arch="convae", width=16, device="auto", **kw):
    """One constructor for every arm, so sweep scripts don't branch on names.

    Placing the module on the device happens HERE rather than at the call site.
    The non-learned arms hold their filters in registered buffers and have no
    parameters, so nothing else ever moves them -- the learned arms are moved
    inside train_recovery/load_recovery, which is why a forgotten .to(DEVICE)
    fails only for box and unsharp, and only once a CUDA batch reaches them.

    device="auto" uses CUDA when available; pass None to leave placement alone.
    """
    if arch in NON_LEARNED:
        cls = NON_LEARNED[arch]
        module = None if cls is None else cls()
    else:
        learned = _learned()
        if arch not in learned:
            raise KeyError(f"unknown arch {arch!r}; have {sorted(ARM_NAMES)}")
        cls = learned[arch]
        if width is None:
            # Published configuration -- what the paper actually reports. Scaled
            # variants are a different claim and must be labelled as such.
            kw = {**getattr(cls, "PUBLISHED", {}), **kw}
            module = cls(**kw)
        else:
            module = cls(width=width, **kw)

    if module is not None and device is not None:
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = module.to(device)
    return module


def count_params(m):
    return 0 if m is None else sum(p.numel() for p in m.parameters())
