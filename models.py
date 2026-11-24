"""
Encoder/decoder architecture for masked-volume self-supervised pretraining.

Architecture note: the JD's SSL fluency bullet (masked image modeling, JEPA,
contrastive) is normally realized with a Vision Transformer for the MAE
recipe. This sandbox has a single CPU core, so a ViT3D (patchify + multi-head
attention over ~hundreds of tokens) would be too slow to train to convergence
in a reasonable time here. Instead this implements the same *pretext task*
(reconstruct masked regions of the volume from context) with a convolutional
encoder-decoder -- i.e. a context-encoder / convolutional-MAE hybrid
(Pathak et al. 2016 style masking + He et al. 2022 MAE-style masked-region-only
loss). The masking strategy, loss, and downstream linear-probe / fine-tune
protocol are the same as MAE; only the backbone differs. Swapping the encoder
for a 3D ViT is a contained change (see README.md) once GPU compute is
available.
"""
import torch
import torch.nn as nn


def conv_block(c_in, c_out, stride=2):
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm3d(c_out),
        nn.GELU(),
    )


def deconv_block(c_in, c_out, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose3d(c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=output_padding),
        nn.BatchNorm3d(c_out),
        nn.GELU(),
    )


class Encoder3D(nn.Module):
    """size 24 -> 12 -> 6 -> 3, channels 1 -> 16 -> 32 -> 64, embed_dim-d pooled embedding."""

    def __init__(self, in_ch=1, base=16, embed_dim=128):
        super().__init__()
        self.stem = conv_block(in_ch, base, stride=2)       # 24 -> 12
        self.stage1 = conv_block(base, base * 2, stride=2)  # 12 -> 6
        self.stage2 = conv_block(base * 2, base * 4, stride=2)  # 6 -> 3
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(base * 4, embed_dim)
        self.feat_ch = base * 4

    def forward(self, x, return_feature_map=False):
        f0 = self.stem(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        pooled = self.pool(f2).flatten(1)
        embed = self.proj(pooled)
        if return_feature_map:
            return embed, f2
        return embed


class Decoder3D(nn.Module):
    """Mirrors Encoder3D: 3 -> 6 -> 12 -> 24, reconstructs the input volume."""

    def __init__(self, base=16):
        super().__init__()
        self.up1 = deconv_block(base * 4, base * 2)  # 3 -> 6
        self.up2 = deconv_block(base * 2, base)      # 6 -> 12
        self.up3 = deconv_block(base, base)          # 12 -> 24
        self.out = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, f2):
        x = self.up1(f2)
        x = self.up2(x)
        x = self.up3(x)
        return torch.sigmoid(self.out(x))


class MaskedAutoencoder3D(nn.Module):
    def __init__(self, base=16, embed_dim=128):
        super().__init__()
        self.encoder = Encoder3D(base=base, embed_dim=embed_dim)
        self.decoder = Decoder3D(base=base)

    def forward(self, x_masked):
        _, f2 = self.encoder(x_masked, return_feature_map=True)
        recon = self.decoder(f2)
        return recon


class ClassifierHead(nn.Module):
    def __init__(self, embed_dim=128, num_classes=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, embed):
        return self.net(embed)


class EncoderClassifier(nn.Module):
    """Wraps an Encoder3D + ClassifierHead; encoder can be frozen or fine-tuned."""

    def __init__(self, encoder, num_classes=6, embed_dim=128, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.head = ClassifierHead(embed_dim=embed_dim, num_classes=num_classes)
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                embed = self.encoder(x)
        else:
            embed = self.encoder(x)
        return self.head(embed)


def random_cube_mask(x, patch=4, mask_ratio=0.5):
    """Zero out random non-overlapping cubic patches of the input volume.
    Returns (masked_x, mask) where mask is 1 where MASKED (loss is computed
    only on these voxels, as in MAE)."""
    B, C, D, H, W = x.shape
    assert D % patch == 0 and H % patch == 0 and W % patch == 0
    nd, nh, nw = D // patch, H // patch, W // patch
    n_patches = nd * nh * nw
    n_mask = max(1, int(round(n_patches * mask_ratio)))

    mask_full = torch.zeros(B, 1, D, H, W, device=x.device)
    for b in range(B):
        idx = torch.randperm(n_patches, device=x.device)[:n_mask]
        for flat in idx.tolist():
            di, hi, wi = flat // (nh * nw), (flat // nw) % nh, flat % nw
            mask_full[b, 0,
                      di * patch:(di + 1) * patch,
                      hi * patch:(hi + 1) * patch,
                      wi * patch:(wi + 1) * patch] = 1.0

    x_masked = x * (1 - mask_full)
    return x_masked, mask_full
