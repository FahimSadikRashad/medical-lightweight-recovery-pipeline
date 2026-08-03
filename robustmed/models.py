"""Classifier, recovery autoencoder, and the non-learned control."""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def backbone(n_classes, pretrained=True):
    """ImageNet MobileNetV2 with a fresh head.

    Defined once. The original notebook redefined this five times -- that is
    how the two baselines silently drift apart.
    """
    from torchvision import models
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m


class Classifier(nn.Module):
    """Takes [0,1] tensors and normalizes internally.

    Keeping normalization here is what lets one loader feed both the AE
    (pixel space) and the classifier.
    """

    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.net = backbone(n_classes, pretrained)
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


class ConvAE(nn.Module):
    """Recovery module. width scales capacity (w=4 is ~1k params).

    residual=True predicts clean as input+correction, which protects clean
    inputs from being smoothed. Reported results use residual=False --
    check docs/FINDINGS.md before flipping it.
    """

    def __init__(self, width=16, residual=False):
        super().__init__()
        w = width
        self.width, self.residual = width, residual
        self.enc = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1), nn.ReLU(inplace=True),        # 64 -> 32
            nn.Conv2d(w, w * 2, 3, 2, 1), nn.ReLU(inplace=True),    # 32 -> 16
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w * 2, w, 4, 2, 1), nn.ReLU(inplace=True),  # 16 -> 32
            nn.ConvTranspose2d(w, 3, 4, 2, 1),                             # 32 -> 64
        )

    def forward(self, x):
        out = self.dec(self.enc(x))
        if self.residual:
            out = x + out
        return torch.clamp(out, 0.0, 1.0)


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


def count_params(m):
    return 0 if m is None else sum(p.numel() for p in m.parameters())
