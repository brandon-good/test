"""film_emulation_impl.py — Cleaned, deduplicated, and fixed

Merged the duplicated training blocks into a single robust training loop,
fixed naming collisions, removed broken prints, and ensured the code is
syntactically correct and consistent.
"""

# ============================= Imports =============================
import os
import argparse
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as Ff  # functional API renamed to avoid collision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

# LPIPS for perceptual similarity (optional)
try:
    import lpips

    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. Install with: pip install lpips")
    print("LPIPS metrics will be disabled.")


# ====================================================================
#                           DATASETS
# ====================================================================


class ResizePreserveAR:
    """Resize so that the LONGER side == max_size while preserving aspect ratio.
    Then pad to a square so that batching still works cleanly.
    """

    def __init__(self, max_size=512, fill=0):
        self.max_size = max_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        pad_w = self.max_size - new_w
        pad_h = self.max_size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img = transforms.functional.pad(img, (left, top, right, bottom), fill=self.fill)
        return img


class UnpairedImageFolder(Dataset):
    """Loads images from a single folder with optional augmentation."""

    def __init__(self, root, transform=None, augment=False, max_size=None):
        self.root = Path(root)
        self.files = sorted(
            [
                p
                for p in self.root.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if max_size is not None:
            self.files = self.files[:max_size]
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root}")

        self.transform = transform
        self.augment = augment

        if augment:
            self.aug_transform = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")

        if self.augment:
            img = self.aug_transform(img)

        if self.transform:
            img = self.transform(img)
        return img


# ====================================================================
#                           MODEL (Generators)
# ====================================================================


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1):
        super().__init__()
        pad = ks // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PerPixelAffineNet(nn.Module):
    """U-Net-lite generator that outputs a 12-channel per-pixel affine map.

    The 12 channels encode a 3x3 matrix (9 values) and a 3-vector bias per pixel.
    """

    def __init__(self, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(3, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = nn.Sequential(
            ConvBlock(base_ch * 8, base_ch * 8),
            ConvBlock(base_ch * 8, base_ch * 8),
        )

        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 12, kernel_size=1)

        # Initialize output layer to near-identity transform
        with torch.no_grad():
            self.out_conv.weight.zero_()
            identity_bias = torch.tensor(
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32
            )
            if self.out_conv.bias is not None:
                self.out_conv.bias.copy_(identity_bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(Ff.avg_pool2d(e1, 2))
        e3 = self.enc3(Ff.avg_pool2d(e2, 2))
        e4 = self.enc4(Ff.avg_pool2d(e3, 2))

        b = self.bottleneck(e4)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        out = self.out_conv(d2)
        return out


# ====================================================================
#                           DISCRIMINATORS
# ====================================================================


class PatchDiscriminator(nn.Module):
    """PatchGAN with spectral normalization for training stability."""

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        layers = []
        layers += [
            nn.utils.spectral_norm(nn.Conv2d(in_ch, base_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers += [
            nn.utils.spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)),
            nn.InstanceNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers += [
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)),
            nn.InstanceNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers += [nn.Conv2d(base_ch * 4, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ====================================================================
#                 HELPER: APPLY AFFINE TRANSFORM TO IMAGE
# ====================================================================


def apply_perpixel_affine(
    transform_map: torch.Tensor, inp: torch.Tensor
) -> torch.Tensor:
    """
    transform_map: (B, 12, H, W)
    inp: (B, 3, H, W)
    Returns: transformed rgb image (B,3,H,W)
    """
    B, C, H, W = transform_map.shape
    assert C == 12
    M = transform_map[:, :9, :, :].view(B, 3, 3, H, W)
    b = transform_map[:, 9:, :, :].view(B, 3, 1, H, W)
    inp_exp = inp.view(B, 3, 1, H, W)
    out = (M * inp_exp).sum(dim=1) + b.squeeze(2)
    return out


# ====================================================================
#                       FILM GRAIN MODEL (IMPROVED)
# ====================================================================


class MultiScaleGrainEstimator(nn.Module):
    """Multi-scale grain estimator that captures both fine and coarse grain patterns.

    This better mimics real film grain which has structure at multiple scales.
    """

    def __init__(self, in_ch=3, out_ch=1, base_ch=32):
        super().__init__()

        # Fine grain branch (high frequency)
        self.fine_grain = nn.Sequential(
            nn.Conv2d(in_ch + 1, base_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 1),
        )

        # Coarse grain branch (lower frequency, larger structures)
        self.coarse_grain = nn.Sequential(
            nn.Conv2d(in_ch + 1, base_ch, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 1),
        )

        # Learnable blend weight
        self.blend = nn.Parameter(torch.tensor(0.7))  # Start favoring fine grain

    def forward(self, x):
        # Compute luminance channel
        luminance = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Grain is stronger in darker regions (inverse relationship with light)
        grain_weight = (1.0 - luminance).clamp(0.1, 1.0)  # Prevent complete suppression

        inp = torch.cat([x, luminance], dim=1)

        fine = self.fine_grain(inp)
        coarse = self.coarse_grain(inp)

        # Blend fine and coarse grain
        blend_weight = torch.sigmoid(self.blend)
        grain = blend_weight * fine + (1 - blend_weight) * coarse

        return grain * grain_weight


# ====================================================================
#                           LOSSES
# ====================================================================

adv_criterion = nn.MSELoss()  # LSGAN loss - more stable than BCE
l1_criterion = nn.L1Loss()


class LPIPSMetric:
    """LPIPS perceptual similarity metric wrapper."""

    def __init__(self, net="alex", device="cpu"):
        if not LPIPS_AVAILABLE:
            raise RuntimeError("lpips package not installed. Run: pip install lpips")

        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()
        for param in self.loss_fn.parameters():
            param.requires_grad = False
        self.device = device

    def compute(self, img1, img2):
        # LPIPS expects images in range [-1, 1]
        img1_normalized = img1 * 2.0 - 1.0
        img2_normalized = img2 * 2.0 - 1.0

        with torch.no_grad():
            distance = self.loss_fn(img1_normalized, img2_normalized)

        return distance.mean().item()

    def compute_batch(self, img1, img2):
        img1_normalized = img1 * 2.0 - 1.0
        img2_normalized = img2 * 2.0 - 1.0

        with torch.no_grad():
            distances = self.loss_fn(img1_normalized, img2_normalized)

        return distances.squeeze().cpu().numpy()


class PerceptualLoss(nn.Module):
    """Multi-layer perceptual loss for better feature matching."""

    def __init__(self, device="cpu"):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval().to(device)

        # Extract features from multiple layers
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, a, b):
        a_norm = (a - self.mean) / self.std
        b_norm = (b - self.mean) / self.std

        a_1 = self.slice1(a_norm)
        b_1 = self.slice1(b_norm)
        loss1 = self.criterion(a_1, b_1)

        a_2 = self.slice2(a_1)
        b_2 = self.slice2(b_1)
        loss2 = self.criterion(a_2, b_2)

        a_3 = self.slice3(a_2)
        b_3 = self.slice3(b_2)
        loss3 = self.criterion(a_3, b_3)

        return loss1 + loss2 + loss3


# ====================================================================
#                      EMA FOR STABLE INFERENCE
# ====================================================================


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])


# ====================================================================
#                      TRAINING (Unpaired Cycle)
# ====================================================================


def train_unpaired(
    data_root,
    epochs=20,
    batch_size=4,
    lr=2e-4,
    device="mps",
    lambda_cycle=3.0,
    lambda_id=0.5,
    use_perc=True,
    use_augment=True,
    val_interval=5,
    compute_lpips=True,
):
    """Enhanced unpaired training with validation and LPIPS metrics."""

    if torch.backends.mps.is_available() and device == "mps":
        device = torch.device("mps")
    elif torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    transform = transforms.Compose(
        [ResizePreserveAR(max_size=512), transforms.ToTensor()]
    )

    digital_ds = UnpairedImageFolder(
        os.path.join(data_root, "digi"), transform=transform, augment=use_augment
    )
    film_ds = UnpairedImageFolder(
        os.path.join(data_root, "film"), transform=transform, augment=use_augment
    )

    # Split into train/val (90/10)
    digital_train_size = int(0.9 * len(digital_ds))
    digital_val_size = len(digital_ds) - digital_train_size
    digital_train, digital_val = torch.utils.data.random_split(
        digital_ds, [digital_train_size, digital_val_size]
    )

    film_train_size = int(0.9 * len(film_ds))
    film_val_size = len(film_ds) - film_train_size
    film_train, film_val = torch.utils.data.random_split(
        film_ds, [film_train_size, film_val_size]
    )

    digital_loader = DataLoader(
        digital_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    film_loader = DataLoader(
        film_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    digital_val_loader = DataLoader(digital_val, batch_size=1, shuffle=False)
    film_val_loader = DataLoader(film_val, batch_size=1, shuffle=False)

    # Initialize models
    G = PerPixelAffineNet(base_ch=32).to(device)
    H = PerPixelAffineNet(base_ch=32).to(
        device
    )  # previously named F; renamed to H to avoid Ff collision
    GrainNet = MultiScaleGrainEstimator().to(device)
    D_film = PatchDiscriminator().to(device)
    D_dig = PatchDiscriminator().to(device)

    # EMA
    ema_G = EMA(G, decay=0.999)
    ema_H = EMA(H, decay=0.999)

    # Optimizers
    g_optimizer = torch.optim.AdamW(
        list(G.parameters()) + list(H.parameters()) + list(GrainNet.parameters()),
        lr=lr,
        betas=(0.5, 0.999),
        weight_decay=1e-4,
    )
    d_optimizer = torch.optim.AdamW(
        list(D_film.parameters()) + list(D_dig.parameters()),
        lr=lr,
        betas=(0.5, 0.999),
        weight_decay=1e-4,
    )

    # Schedulers
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        g_optimizer, T_max=max(1, epochs)
    )
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        d_optimizer, T_max=max(1, epochs)
    )

    perc = PerceptualLoss(device=device) if use_perc else None

    lpips_metric = None
    if compute_lpips and LPIPS_AVAILABLE:
        try:
            lpips_metric = LPIPSMetric(net="alex", device=device)
            print("✓ LPIPS metric initialized (AlexNet backbone)")
        except Exception as e:
            print(f"Warning: Could not initialize LPIPS: {e}")
            lpips_metric = None

    history = {
        "g_loss": [],
        "d_loss": [],
        "cycle_loss": [],
        "id_loss": [],
        "val_cycle_loss": [],
        "val_lpips_dig2film": [],
        "val_lpips_film2dig": [],
        "val_lpips_cycle_dig": [],
        "val_lpips_cycle_film": [],
    }

    best_val_loss = float("inf")
    best_lpips_score = float("inf")
    patience_counter = 0
    patience_limit = 10

    num_batches = min(len(digital_loader), len(film_loader))

    for epoch in range(epochs):
        G.train()
        H.train()
        GrainNet.train()
        D_film.train()
        D_dig.train()

        dig_iter = iter(digital_loader)
        film_iter = iter(film_loader)

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_cycle_loss = 0.0

        for it in range(num_batches):
            # Load batches
            try:
                real_dig = next(dig_iter)
            except StopIteration:
                dig_iter = iter(digital_loader)
                real_dig = next(dig_iter)

            try:
                real_film = next(film_iter)
            except StopIteration:
                film_iter = iter(film_loader)
                real_film = next(film_iter)
            # to lower "identity" learning
            real_dig = real_dig.to(device)
            real_dig += 0.02 * torch.randn_like(real_dig)
            real_film = real_film.to(device)
            real_film += 0.02 * torch.randn_like(real_film)

            # ---------- Discriminator step ----------
            # Generate
            tmap_g = G(real_dig)
            fake_film_base = torch.clamp(
                apply_perpixel_affine(tmap_g, real_dig), 0.0, 1.0
            )
            grain_map = GrainNet(fake_film_base)
            fake_film = torch.clamp(fake_film_base + 0.05 * grain_map, 0.0, 1.0)

            tmap_h = H(real_film)
            fake_dig = torch.clamp(apply_perpixel_affine(tmap_h, real_film), 0.0, 1.0)

            # Discriminator losses
            d_optimizer.zero_grad()

            pred_real_film = D_film(real_film)
            loss_d_real_film = adv_criterion(
                pred_real_film, torch.ones_like(pred_real_film)
            )
            pred_fake_film = D_film(fake_film.detach())
            loss_d_fake_film = adv_criterion(
                pred_fake_film, torch.zeros_like(pred_fake_film)
            )

            pred_real_dig = D_dig(real_dig)
            loss_d_real_dig = adv_criterion(
                pred_real_dig, torch.ones_like(pred_real_dig)
            )
            pred_fake_dig = D_dig(fake_dig.detach())
            loss_d_fake_dig = adv_criterion(
                pred_fake_dig, torch.zeros_like(pred_fake_dig)
            )

            loss_d = 0.5 * (
                loss_d_real_film + loss_d_fake_film + loss_d_real_dig + loss_d_fake_dig
            )
            loss_d.backward()
            d_optimizer.step()

            # ---------- Generator step ----------
            g_optimizer.zero_grad()

            # Adversarial loss
            pred_fake_film_for_g = D_film(fake_film)
            loss_g_adv_film = adv_criterion(
                pred_fake_film_for_g, torch.ones_like(pred_fake_film_for_g)
            )

            pred_fake_dig_for_h = D_dig(fake_dig)
            loss_h_adv_dig = adv_criterion(
                pred_fake_dig_for_h, torch.ones_like(pred_fake_dig_for_h)
            )

            # Cycle consistency
            rec_tmap = H(fake_film)
            rec_dig = torch.clamp(apply_perpixel_affine(rec_tmap, fake_film), 0.0, 1.0)
            loss_cycle_dig = l1_criterion(rec_dig, real_dig)

            rec_tmap2 = G(fake_dig)
            rec_film_base = torch.clamp(
                apply_perpixel_affine(rec_tmap2, fake_dig), 0.0, 1.0
            )
            rec_grain = GrainNet(rec_film_base)
            rec_film = torch.clamp(rec_film_base + 0.05 * rec_grain, 0.0, 1.0)
            loss_cycle_film = l1_criterion(rec_film, real_film)

            loss_cycle = 0.5 * (loss_cycle_dig + loss_cycle_film)

            # Identity losses
            id_tmap_g = G(real_film)
            id_g = torch.clamp(apply_perpixel_affine(id_tmap_g, real_film), 0.0, 1.0)
            loss_id_g = l1_criterion(id_g, real_film)

            id_tmap_h = H(real_dig)
            id_h = torch.clamp(apply_perpixel_affine(id_tmap_h, real_dig), 0.0, 1.0)
            loss_id_h = l1_criterion(id_h, real_dig)

            loss_id = 0.5 * (loss_id_g + loss_id_h)

            # Smoothness regularizer
            def smoothness_loss(tmap):
                dx = torch.abs(tmap[:, :, :-1, :] - tmap[:, :, 1:, :])
                dy = torch.abs(tmap[:, :, :, :-1] - tmap[:, :, :, 1:])
                return dx.mean() + dy.mean()

            loss_smooth = smoothness_loss(tmap_g) + smoothness_loss(tmap_h)

            # Perceptual loss
            loss_perc = torch.tensor(0.0, device=device)
            if perc is not None:
                loss_perc = 0.5 * (
                    perc(fake_film, real_film) + perc(fake_dig, real_dig)
                )

            # Grain frequency penalty
            grain_freq_penalty = torch.mean(Ff.avg_pool2d(grain_map.abs(), 4) ** 2)

            loss_g_total = (
                loss_g_adv_film
                + loss_h_adv_dig
                + lambda_cycle * loss_cycle
                + lambda_id * loss_id
                + 0.1 * loss_perc
                + 0.005 * loss_smooth
                + 0.02 * grain_freq_penalty
            )

            loss_g_total.backward()

            # Gradient clipping and step
            torch.nn.utils.clip_grad_norm_(
                list(G.parameters())
                + list(H.parameters())
                + list(GrainNet.parameters()),
                max_norm=1.0,
            )
            g_optimizer.step()

            # Update EMA
            ema_G.update()
            ema_H.update()

            # Track losses
            epoch_g_loss += loss_g_total.item()
            epoch_d_loss += loss_d.item()
            epoch_cycle_loss += loss_cycle.item()

            if (it + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] Iter [{it+1}/{num_batches}] | D: {loss_d.item():.4f} | G: {loss_g_total.item():.4f} | Cycle: {loss_cycle.item():.4f} | ID: {loss_id.item():.4f}"
                )

        # End batches
        g_scheduler.step()
        d_scheduler.step()

        history["g_loss"].append(epoch_g_loss / float(num_batches))
        history["d_loss"].append(epoch_d_loss / float(num_batches))
        history["cycle_loss"].append(epoch_cycle_loss / float(num_batches))

        # Validation
        if (epoch + 1) % val_interval == 0:
            G.eval()
            H.eval()
            GrainNet.eval()

            val_cycle_loss = 0.0
            lpips_dig2film_scores = []
            lpips_film2dig_scores = []
            lpips_cycle_dig_scores = []
            lpips_cycle_film_scores = []

            with torch.no_grad():
                val_pairs = list(zip(digital_val_loader, film_val_loader))
                for val_dig, val_film in val_pairs:
                    val_dig = val_dig.to(device)
                    val_film = val_film.to(device)

                    tmap_g = G(val_dig)
                    fake_film_base = torch.clamp(
                        apply_perpixel_affine(tmap_g, val_dig), 0.0, 1.0
                    )
                    grain = GrainNet(fake_film_base)
                    fake_film = torch.clamp(fake_film_base + 0.05 * grain, 0.0, 1.0)

                    tmap_h = H(val_film)
                    fake_dig = torch.clamp(
                        apply_perpixel_affine(tmap_h, val_film), 0.0, 1.0
                    )

                    # Cycles
                    rec_tmap = H(fake_film)
                    rec_dig = torch.clamp(
                        apply_perpixel_affine(rec_tmap, fake_film), 0.0, 1.0
                    )
                    val_cycle_loss += l1_criterion(rec_dig, val_dig).item()

                    rec_tmap2 = G(fake_dig)
                    rec_film_base = torch.clamp(
                        apply_perpixel_affine(rec_tmap2, fake_dig), 0.0, 1.0
                    )
                    rec_grain = GrainNet(rec_film_base)
                    rec_film = torch.clamp(rec_film_base + 0.05 * rec_grain, 0.0, 1.0)

                    if lpips_metric is not None:
                        lpips_d2f = lpips_metric.compute(val_dig, fake_film)
                        lpips_dig2film_scores.append(lpips_d2f)

                        lpips_f2d = lpips_metric.compute(val_film, fake_dig)
                        lpips_film2dig_scores.append(lpips_f2d)

                        lpips_cycle_d = lpips_metric.compute(val_dig, rec_dig)
                        lpips_cycle_dig_scores.append(lpips_cycle_d)

                        lpips_cycle_f = lpips_metric.compute(val_film, rec_film)
                        lpips_cycle_film_scores.append(lpips_cycle_f)

            n_val = max(1, len(digital_val_loader))
            val_cycle_loss /= n_val
            history["val_cycle_loss"].append(val_cycle_loss)

            print("=" * 70)
            print(f"VALIDATION - Epoch {epoch+1}")
            print("=" * 70)
            print(f"Cycle Loss (L1): {val_cycle_loss:.4f}")

            if lpips_metric is not None and len(lpips_dig2film_scores) > 0:
                avg_lpips_d2f = float(np.mean(lpips_dig2film_scores))
                avg_lpips_f2d = float(np.mean(lpips_film2dig_scores))
                avg_lpips_cycle_d = float(np.mean(lpips_cycle_dig_scores))
                avg_lpips_cycle_f = float(np.mean(lpips_cycle_film_scores))

                history["val_lpips_dig2film"].append(avg_lpips_d2f)
                history["val_lpips_film2dig"].append(avg_lpips_f2d)
                history["val_lpips_cycle_dig"].append(avg_lpips_cycle_d)
                history["val_lpips_cycle_film"].append(avg_lpips_cycle_f)

                print("LPIPS Perceptual Similarity (lower = more similar):")
                print(f"  Digital -> Film:        {avg_lpips_d2f:.4f}")
                print(f"  Film -> Digital:        {avg_lpips_f2d:.4f}")
                print(
                    f"  Cycle (Digital):        {avg_lpips_cycle_d:.4f} ← (should be low)"
                )
                print(
                    f"  Cycle (Film):           {avg_lpips_cycle_f:.4f} ← (should be low)"
                )

                combined_lpips = (avg_lpips_cycle_d + avg_lpips_cycle_f) * 2.0
                print(f"Combined LPIPS Score:    {combined_lpips:.4f}")

                if combined_lpips < best_lpips_score:
                    best_lpips_score = combined_lpips
                    print("✓ New best LPIPS score based on cycle consistency")

            print("=" * 70 + "")

            # Save best L1 cycle model
            if val_cycle_loss < best_val_loss:
                best_val_loss = val_cycle_loss
                patience_counter = 0
                ema_G.apply_shadow()
                ema_H.apply_shadow()
                torch.save(G.state_dict(), "best_model_G.pth")
                torch.save(H.state_dict(), "best_model_H.pth")
                torch.save(GrainNet.state_dict(), "best_model_Grain.pth")
                ema_G.restore()
                ema_H.restore()
                print(f"✓ Saved new best models (val cycle loss: {val_cycle_loss:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                print(
                    f"Early stopping at epoch {epoch+1} (no improvement for {patience_limit} validations)"
                )
                break

        # End epoch
        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "G": G.state_dict(),
                    "H": H.state_dict(),
                    "GrainNet": GrainNet.state_dict(),
                    "g_opt": g_optimizer.state_dict(),
                    "d_opt": d_optimizer.state_dict(),
                    "history": history,
                },
                f"checkpoint_epoch_{epoch+1}.pth",
            )

    # Save training history
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("✓ Training complete!")
    print(f"Best validation cycle loss: {best_val_loss:.4f}")
    if lpips_metric is not None:
        print(f"Best LPIPS cycle score: {best_lpips_score:.4f}")


# ====================================================================
#                           EVALUATION
# ====================================================================


def apply_trained_generator(
    model_path, grain_path, src_folder, out_folder, direction="G", device="mps"
):
    """Apply trained generator with grain to all images in src_folder."""
    device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    inv_to_pil = transforms.ToPILImage()

    model = PerPixelAffineNet(base_ch=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    grain_net = MultiScaleGrainEstimator().to(device)
    if os.path.exists(grain_path):
        grain_net.load_state_dict(torch.load(grain_path, map_location=device))
    grain_net.eval()

    p = Path(src_folder)
    out_p = Path(out_folder)
    out_p.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(
        [q for q in p.iterdir() if q.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    ):
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Pad to square
        max_dim = max(orig_w, orig_h)
        pad_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        offset = ((max_dim - orig_w) // 2, (max_dim - orig_h) // 2)
        pad_img.paste(img, offset)

        pad_img_resized = pad_img.resize((512, 512), Image.BICUBIC)
        inp = transforms.ToTensor()(pad_img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            tmap = model(inp)
            out_base = torch.clamp(apply_perpixel_affine(tmap, inp), 0, 1)

            # Add grain if requested
            if direction == "G":
                grain = grain_net(out_base)
                out = torch.clamp(out_base + 0.05 * grain, 0, 1)
            else:
                out = out_base

        out_img = inv_to_pil(out.squeeze(0).cpu())

        # Crop back to original aspect ratio
        scale = 512 / max_dim
        left = int(offset[0] * scale)
        top = int(offset[1] * scale)
        right = int(left + orig_w * scale)
        bottom = int(top + orig_h * scale)
        out_img = out_img.crop((left, top, right, bottom))
        out_img = out_img.resize((orig_w, orig_h), Image.BICUBIC)

        out_img.save(out_p / img_path.name, quality=95)
        print(f"Processed: {img_path.name}")

    print(f"✓ Saved outputs to {out_p}")


# ====================================================================
#                    LPIPS EVALUATION (Standalone)
# ====================================================================


def evaluate_lpips_on_folder(
    model_g_path,
    model_f_path,
    grain_path,
    digital_folder,
    film_folder,
    device="mps",
    save_comparisons=True,
    output_folder="./lpips_evaluation",
):
    if not LPIPS_AVAILABLE:
        print("Error: lpips package not installed. Run: pip install lpips")
        return

    device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load models
    G = PerPixelAffineNet(base_ch=32).to(device)
    G.load_state_dict(torch.load(model_g_path, map_location=device))
    G.eval()

    H = PerPixelAffineNet(base_ch=32).to(device)
    H.load_state_dict(torch.load(model_f_path, map_location=device))
    H.eval()

    grain_net = MultiScaleGrainEstimator().to(device)
    if os.path.exists(grain_path):
        grain_net.load_state_dict(torch.load(grain_path, map_location=device))
    grain_net.eval()

    lpips_metric = LPIPSMetric(net="alex", device=device)

    transform = transforms.Compose(
        [ResizePreserveAR(max_size=512), transforms.ToTensor()]
    )

    digital_ds = UnpairedImageFolder(digital_folder, transform=transform)
    film_ds = UnpairedImageFolder(film_folder, transform=transform)

    num_samples = min(len(digital_ds), len(film_ds))

    results = {
        "lpips_digital_to_film": [],
        "lpips_film_to_digital": [],
        "lpips_cycle_digital": [],
        "lpips_cycle_film": [],
        "l1_cycle_digital": [],
        "l1_cycle_film": [],
    }

    if save_comparisons:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "digital_to_film").mkdir(exist_ok=True)
        (output_path / "film_to_digital").mkdir(exist_ok=True)
        (output_path / "cycles").mkdir(exist_ok=True)

    inv_to_pil = transforms.ToPILImage()

    print(f"Evaluating {num_samples} image pairs...")

    with torch.no_grad():
        for idx in range(num_samples):
            digital_img = digital_ds[idx].unsqueeze(0).to(device)
            film_img = film_ds[idx].unsqueeze(0).to(device)

            # Digital -> Film
            tmap_g = G(digital_img)
            fake_film_base = torch.clamp(
                apply_perpixel_affine(tmap_g, digital_img), 0.0, 1.0
            )
            grain = grain_net(fake_film_base)
            fake_film = torch.clamp(fake_film_base + 0.05 * grain, 0.0, 1.0)

            # Film -> Digital
            tmap_h = H(film_img)
            fake_digital = torch.clamp(
                apply_perpixel_affine(tmap_h, film_img), 0.0, 1.0
            )

            # Cycles
            rec_tmap_d = H(fake_film)
            rec_digital = torch.clamp(
                apply_perpixel_affine(rec_tmap_d, fake_film), 0.0, 1.0
            )

            rec_tmap_f = G(fake_digital)
            rec_film_base = torch.clamp(
                apply_perpixel_affine(rec_tmap_f, fake_digital), 0.0, 1.0
            )
            rec_grain = grain_net(rec_film_base)
            rec_film = torch.clamp(rec_film_base + 0.05 * rec_grain, 0.0, 1.0)

            # Compute LPIPS
            lpips_d2f = lpips_metric.compute(digital_img, fake_film)
            lpips_f2d = lpips_metric.compute(film_img, fake_digital)
            lpips_cycle_d = lpips_metric.compute(digital_img, rec_digital)
            lpips_cycle_f = lpips_metric.compute(film_img, rec_film)

            # Compute L1 for cycle consistency
            l1_cycle_d = l1_criterion(rec_digital, digital_img).item()
            l1_cycle_f = l1_criterion(rec_film, film_img).item()

            results["lpips_digital_to_film"].append(lpips_d2f)
            results["lpips_film_to_digital"].append(lpips_f2d)
            results["lpips_cycle_digital"].append(lpips_cycle_d)
            results["lpips_cycle_film"].append(lpips_cycle_f)
            results["l1_cycle_digital"].append(l1_cycle_d)
            results["l1_cycle_film"].append(l1_cycle_f)

            # Save visual comparisons (first 20)
            if save_comparisons and idx < 20:
                comparison = torch.cat([digital_img, fake_film, film_img], dim=3)
                inv_to_pil(comparison.squeeze(0).cpu()).save(
                    output_path / "digital_to_film" / f"{idx:03d}_comparison.jpg"
                )

                comparison = torch.cat([film_img, fake_digital, digital_img], dim=3)
                inv_to_pil(comparison.squeeze(0).cpu()).save(
                    output_path / "film_to_digital" / f"{idx:03d}_comparison.jpg"
                )

                cycle_vis = torch.cat(
                    [
                        digital_img,
                        fake_film,
                        rec_digital,
                        film_img,
                        fake_digital,
                        rec_film,
                    ],
                    dim=3,
                )
                inv_to_pil(cycle_vis.squeeze(0).cpu()).save(
                    output_path / "cycles" / f"{idx:03d}_cycles.jpg"
                )

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{num_samples} images")

    # Summarize
    print("" + "=" * 70)
    print("LPIPS EVALUATION RESULTS")
    print("=" * 70)
    print("Perceptual Similarity (LPIPS - lower is better):")
    print(
        f"  Digital -> Film:        {np.mean(results['lpips_digital_to_film']):.4f} ± {np.std(results['lpips_digital_to_film']):.4f}"
    )
    print(
        f"  Film -> Digital:        {np.mean(results['lpips_film_to_digital']):.4f} ± {np.std(results['lpips_film_to_digital']):.4f}"
    )
    print("Cycle Consistency (LPIPS - should be LOW):")
    print(
        f"  Digital cycle:          {np.mean(results['lpips_cycle_digital']):.4f} ± {np.std(results['lpips_cycle_digital']):.4f}"
    )
    print(
        f"  Film cycle:             {np.mean(results['lpips_cycle_film']):.4f} ± {np.std(results['lpips_cycle_film']):.4f}"
    )
    print("Cycle Consistency (L1):")
    print(
        f"  Digital cycle:          {np.mean(results['l1_cycle_digital']):.4f} ± {np.std(results['l1_cycle_digital']):.4f}"
    )
    print(
        f"  Film cycle:             {np.mean(results['l1_cycle_film']):.4f} ± {np.std(results['l1_cycle_film']):.4f}"
    )
    print("=" * 70)

    summary = {
        "num_samples": num_samples,
        "metrics": {
            "lpips_digital_to_film": {
                "mean": float(np.mean(results["lpips_digital_to_film"])),
                "std": float(np.std(results["lpips_digital_to_film"])),
                "min": float(np.min(results["lpips_digital_to_film"])),
                "max": float(np.max(results["lpips_digital_to_film"])),
            },
            "lpips_film_to_digital": {
                "mean": float(np.mean(results["lpips_film_to_digital"])),
                "std": float(np.std(results["lpips_film_to_digital"])),
            },
            "lpips_cycle_digital": {
                "mean": float(np.mean(results["lpips_cycle_digital"])),
                "std": float(np.std(results["lpips_cycle_digital"])),
            },
            "lpips_cycle_film": {
                "mean": float(np.mean(results["lpips_cycle_film"])),
                "std": float(np.std(results["lpips_cycle_film"])),
            },
        },
    }

    if save_comparisons:
        with open(output_path / "lpips_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Results saved to {output_path / 'lpips_results.json'}")
        print(f"✓ Visual comparisons saved to {output_path}")

    return results


# ====================================================================
#                              CLI
# ====================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Film Emulation with CycleGAN-style training and LPIPS evaluation"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./dataset_root",
        help="Root directory with 'digital' and 'film' subdirectories",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train_unpaired",
        choices=["train_unpaired", "apply", "eval_lpips"],
        help="Mode: train_unpaired, apply, or eval_lpips",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default="mps", help="Device: mps, cuda, or cpu"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_model_G.pth",
        help="Path to G model (digital->film) for inference",
    )
    parser.add_argument(
        "--model_f",
        type=str,
        default="best_model_H.pth",
        help="Path to H model (film->digital) for eval_lpips",
    )
    parser.add_argument(
        "--grain_model",
        type=str,
        default="best_model_Grain.pth",
        help="Path to grain model for inference",
    )
    parser.add_argument(
        "--src", type=str, default="./digital", help="Source folder for inference"
    )
    parser.add_argument(
        "--out", type=str, default="./out", help="Output folder for inference"
    )
    parser.add_argument(
        "--no_augment", action="store_true", help="Disable data augmentation"
    )
    parser.add_argument(
        "--no_perc", action="store_true", help="Disable perceptual loss"
    )
    parser.add_argument(
        "--no_lpips", action="store_true", help="Disable LPIPS metrics during training"
    )
    parser.add_argument(
        "--lambda_cycle",
        type=float,
        default=10.0,
        help="Weight for cycle consistency loss",
    )
    parser.add_argument(
        "--lambda_id", type=float, default=5.0, help="Weight for identity loss"
    )
    parser.add_argument(
        "--val_interval", type=int, default=5, help="Validation interval in epochs"
    )
    parser.add_argument(
        "--digital_test",
        type=str,
        default=None,
        help="Digital test folder for LPIPS evaluation",
    )
    parser.add_argument(
        "--film_test",
        type=str,
        default=None,
        help="Film test folder for LPIPS evaluation",
    )

    args = parser.parse_args()

    if args.mode == "train_unpaired":
        train_unpaired(
            data_root=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            device=args.device,
            lambda_cycle=args.lambda_cycle,
            lambda_id=args.lambda_id,
            use_perc=not args.no_perc,
            use_augment=not args.no_augment,
            val_interval=args.val_interval,
            compute_lpips=not args.no_lpips,
        )
    elif args.mode == "apply":
        apply_trained_generator(
            model_path=args.model,
            grain_path=args.grain_model,
            src_folder=args.src,
            out_folder=args.out,
            direction="G",
            device=args.device,
        )
    elif args.mode == "eval_lpips":
        digital_test = args.digital_test or os.path.join(args.data, "digital")
        film_test = args.film_test or os.path.join(args.data, "film")
        evaluate_lpips_on_folder(
            model_g_path=args.model,
            model_f_path=args.model_f,
            grain_path=args.grain_model,
            digital_folder=digital_test,
            film_folder=film_test,
            device=args.device,
            save_comparisons=True,
            output_folder=args.out,
        )
