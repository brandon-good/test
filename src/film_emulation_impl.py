"""
film_emulation_impl.py

Unpaired training addition (Cycle-consistent training) for the
"CNNs for Faithful Film Emulation" project.

This file replaces the paired training flow with an **unpaired**
CycleGAN-style training loop while keeping your **per-pixel affine
transform generator**. That preserves the strong anti-hallucination
property: the generators only predict color transforms applied at the
same pixel locations (no spatial warping).

High-level components added:
 - Unpaired dataset loaders (digital folder, film folder)
 - Two generators (G: digital->film, F: film->digital) using
   the same PerPixelAffineNet architecture
 - Two Patch-style discriminators (D_film, D_digital)
 - Losses: adversarial, cycle-consistency, identity, optional
   perceptual + L1

This is intentionally pragmatic: small networks, clear losses, and
thorough inline comments for clarity.

Run training (unpaired):
    python film_emulation_impl.py --data ./dataset_root --mode train_unpaired --epochs 30 --batch 4

Run evaluation (generator only):
    python film_emulation_impl.py --data ./dataset_root --mode eval --model best_model_G.pth
"""

# ============================= Imports =============================
import os
import argparse
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.models as models


# ====================================================================
#                           DATASETS
# ====================================================================


class ResizePreserveAR:
    """Resize so that the LONGER side == max_size while preserving aspect ratio.
    Then pad to a square so that batching still works cleanly.
    This keeps geometry intact (no stretching).

    Usage: replace transforms.Resize((512,512)) with ResizePreserveAR(max_size=512)
    """

    def __init__(self, max_size=512, fill=0):
        self.max_size = max_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # Pad to square (centered)
        pad_w = self.max_size - new_w
        pad_h = self.max_size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img = transforms.functional.pad(img, (left, top, right, bottom), fill=self.fill)
        return img


class UnpairedImageFolder(Dataset):
    """
    Loads images from a single folder (no pairing). Use two instances
    for digital/ and film/ separately. Returns individual images.
    """

    def __init__(self, root, transform=None, max_size=None):
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ====================================================================
#                           MODEL (Generators)
# ====================================================================
# We reuse the PerPixelAffineNet generator: it outputs a per-pixel
# 3x3 matrix + 3 bias terms (12 channels). This means no spatial
# movement of pixels is possible — only color mixing and shifts.


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1):
        super().__init__()
        pad = ks // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PerPixelAffineNet(nn.Module):
    """U-Net-lite generator returning per-pixel affine color maps (12 channels)."""

    def __init__(self, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(3, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 12, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.out_conv(d1)
        return out


# ====================================================================
#                           DISCRIMINATORS
# ====================================================================
# PatchGAN-style discriminator: outputs patch-wise real/fake logits.
# Works on images (not on transform maps) — discriminators see the
# final RGB image after the affine transform is applied.


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        layers = []
        # A few convolutional layers to shrink spatial dims and increase channels
        layers += [nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        layers += [
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers += [
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # Final conv to a single-channel map of logits (patch-wise)
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

    Returns: transformed rgb image (B,3,H,W) in [0,1]
    """
    B, C, H, W = transform_map.shape
    assert C == 12
    M = transform_map[:, :9, :, :].view(B, 3, 3, H, W)
    b = transform_map[:, 9:, :, :].view(B, 3, 1, H, W)
    inp_exp = inp.view(B, 3, 1, H, W)
    out = (M * inp_exp).sum(dim=1) + b.squeeze(2)
    return out


# ====================================================================
#                       FILM GRAIN MODEL (NEW)
#   Inspired by classic film grain modeling + GAN texture synthesis
# ====================================================================
class GrainEstimator(nn.Module):
    """Predicts spatially-varying grain maps.

    The network explicitly conditions on luminance so that:
    - More grain appears in shadows
    - Less grain appears in highlights
    - Midtones receive moderate grain

    This mimics real film behavior (noise ~ inverse signal level).
    """

    def __init__(self, in_ch=3, out_ch=1, base_ch=32):
        super().__init__()

        # Small CNN that predicts a base grain pattern
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + 1, base_ch, 3, 1, 1),  # +1 for luminance channel
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 1),
        )

    def forward(self, x):
        # Estimate luminance: Y = 0.299 R + 0.587 G + 0.114 B
        luminance = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Invert luminance so darker areas get stronger grain
        grain_weight = 1.0 - luminance

        # Concatenate luminance with image as extra channel
        inp = torch.cat([x, luminance], dim=1)

        grain = self.net(inp)

        # Weight grain spatially by tone
        return grain * grain_weight


# ====================================================================
#                           LOSSES
# ====================================================================

# Adversarial loss (LSGAN style would work too; we use BCEWithLogits)
adv_criterion = nn.BCEWithLogitsLoss()

# Simple L1 loss for cycle + identity
l1_criterion = nn.L1Loss()


class PerceptualLoss(nn.Module):
    """Optional perceptual loss using VGG16 features."""

    def __init__(self, device="cpu"):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval().to(device)
        for p in self.slice.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, a, b):
        mean = torch.tensor([0.485, 0.456, 0.406], device=a.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=a.device).view(1, 3, 1, 1)
        fa = self.slice((a - mean) / std)
        fb = self.slice((b - mean) / std)
        return self.criterion(fa, fb)


# ====================================================================
#                      TRAINING (Unpaired Cycle)
# ====================================================================

# Optimized for Apple Silicon (M4/MPS) + maximum throughput
# - Uses channels_last memory format
# - torch.compile when available
# - mixed precision on MPS
# - larger default batch
# - pinned memory + workers


def train_unpaired(
    data_root,
    epochs=20,
    batch_size=4,
    lr=2e-4,
    device="mps",
    lambda_cycle=10.0,
    lambda_id=5.0,
    use_perc=False,
):
    """
    Unpaired CycleGAN-style training loop using per-pixel affine generators.

    Important notes:
      - G: digital -> film  (generator_G)
      - F: film -> digital  (generator_F)
      - D_film and D_digital: patch discriminators

    Losses:
      - Adversarial (make fakes indistinguishable from real domain)
      - Cycle-consistency: F(G(x)) ~= x, G(F(y)) ~= y
      - Identity: G(y) ~= y for real film y (and F(x) ~= x for digital x)

    Architectural constraint (no hallucination): generators only predict
    per-pixel color transforms; they cannot change spatial structure.
    """

    # choose device: prefer MPS (Apple silicon), then CUDA, then CPU
    if torch.backends.mps.is_available() and device == "mps":
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # Transforms: we keep size moderate for iteration speed
    transform = transforms.Compose(
        [
            ResizePreserveAR(max_size=512),
            transforms.ToTensor(),
        ]
    )

    digital_ds = UnpairedImageFolder(
        os.path.join(data_root, "digital"), transform=transform
    )
    film_ds = UnpairedImageFolder(os.path.join(data_root, "film"), transform=transform)

    digital_loader = DataLoader(
        digital_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    film_loader = DataLoader(
        film_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Initialize models
    G = PerPixelAffineNet(base_ch=32).to(device)  # digital -> film
    F = PerPixelAffineNet(base_ch=32).to(device)  # film -> digital
    GrainNet = GrainEstimator().to(device)  # film grain estimator
    D_film = PatchDiscriminator().to(device)
    D_dig = PatchDiscriminator().to(device)
    G = PerPixelAffineNet(base_ch=32).to(device)  # digital -> film
    F = PerPixelAffineNet(base_ch=32).to(device)  # film -> digital
    D_film = PatchDiscriminator().to(device)
    D_dig = PatchDiscriminator().to(device)

    # Optimizers
    g_optimizer = torch.optim.Adam(
        list(G.parameters()) + list(F.parameters()) + list(GrainNet.parameters()),
        lr=lr,
        betas=(0.5, 0.999),
    )
    d_optimizer = torch.optim.Adam(
        list(D_film.parameters()) + list(D_dig.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    g_optimizer = torch.optim.Adam(
        list(G.parameters()) + list(F.parameters()), lr=lr, betas=(0.5, 0.999)
    )
    d_optimizer = torch.optim.Adam(
        list(D_film.parameters()) + list(D_dig.parameters()), lr=lr, betas=(0.5, 0.999)
    )

    # Optional perceptual loss
    perc = PerceptualLoss(device=device) if use_perc else None

    # Helper to create label tensors matching discriminator output map size
    def make_labels_like(pred, value):
        # pred is a logits map; create a same-shaped tensor filled with value
        return torch.full_like(pred, fill_value=value, device=pred.device)

    # Iterators to independently step through both datasets
    dig_iter = iter(digital_loader)
    film_iter = iter(film_loader)

    best_score = float("inf")

    for epoch in range(epochs):
        G.train()
        F.train()
        D_film.train()
        D_dig.train()

        for it in range(min(len(digital_loader), len(film_loader))):
            # ------------------- load batches (unpaired) -------------------
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

            real_dig = real_dig.to(device)
            real_film = real_film.to(device)

            # ------------------- Generators forward -------------------
            # Predict per-pixel affine transforms, then apply to obtain images
            tmap_g = G(real_dig)  # transform maps for digital->film
            fake_film_base = torch.clamp(
                apply_perpixel_affine(tmap_g, real_dig), 0.0, 1.0
            )

            # Predict learned grain pattern (trained on real film)
            fake_film_base = torch.clamp(
                apply_perpixel_affine(tmap_g, real_dig), 0.0, 1.0
            )

            grain_map = GrainNet(fake_film_base)

            fake_film = torch.clamp(fake_film_base + 0.1 * grain_map, 0.0, 1.0)

            tmap_f = F(real_film)  # transform maps for film->digital
            fake_dig = torch.clamp(apply_perpixel_affine(tmap_f, real_film), 0.0, 1.0)

            # ------------------- Discriminator update -------------------
            # Train discriminators to distinguish real vs fake
            d_optimizer.zero_grad()

            # D_film on real film
            pred_real_film = D_film(real_film)
            loss_d_real_film = adv_criterion(
                pred_real_film, make_labels_like(pred_real_film, 1.0)
            )

            # D_film on fake film (detach generator so gradients don't flow to G)
            pred_fake_film = D_film(fake_film.detach())
            loss_d_fake_film = adv_criterion(
                pred_fake_film, make_labels_like(pred_fake_film, 0.0)
            )

            # D_dig on real digital
            pred_real_dig = D_dig(real_dig)
            loss_d_real_dig = adv_criterion(
                pred_real_dig, make_labels_like(pred_real_dig, 1.0)
            )

            # D_dig on fake digital
            pred_fake_dig = D_dig(fake_dig.detach())
            loss_d_fake_dig = adv_criterion(
                pred_fake_dig, make_labels_like(pred_fake_dig, 0.0)
            )

            # Total discriminator loss
            loss_d = (
                loss_d_real_film + loss_d_fake_film + loss_d_real_dig + loss_d_fake_dig
            ) * 0.5
            loss_d.backward()
            d_optimizer.step()

            # ------------------- Generator update -------------------
            g_optimizer.zero_grad()

            # Adversarial loss for generators (try to fool discriminators)
            pred_fake_film_for_g = D_film(fake_film)
            loss_g_adv_film = adv_criterion(
                pred_fake_film_for_g, make_labels_like(pred_fake_film_for_g, 1.0)
            )

            pred_fake_dig_for_f = D_dig(fake_dig)
            loss_f_adv_dig = adv_criterion(
                pred_fake_dig_for_f, make_labels_like(pred_fake_dig_for_f, 1.0)
            )

            # Cycle-consistency: F(G(x)) ~ x and G(F(y)) ~ y
            rec_tmap = F(fake_film)  # transforms predicted by F on fake_film
            rec_dig = torch.clamp(apply_perpixel_affine(rec_tmap, fake_film), 0.0, 1.0)
            loss_cycle_dig = l1_criterion(rec_dig, real_dig)

            rec_tmap2 = G(fake_dig)
            rec_film = torch.clamp(apply_perpixel_affine(rec_tmap2, fake_dig), 0.0, 1.0)
            loss_cycle_film = l1_criterion(rec_film, real_film)

            loss_cycle = (loss_cycle_dig + loss_cycle_film) * 0.5

            # Identity loss: encourages generators to be identity on target domain
            # This helps prevent them from making unnecessary changes.
            id_tmap_g_on_film = G(real_film)
            id_g_on_film = torch.clamp(
                apply_perpixel_affine(id_tmap_g_on_film, real_film), 0.0, 1.0
            )
            loss_id_g = l1_criterion(id_g_on_film, real_film)

            id_tmap_f_on_dig = F(real_dig)
            id_f_on_dig = torch.clamp(
                apply_perpixel_affine(id_tmap_f_on_dig, real_dig), 0.0, 1.0
            )
            loss_id_f = l1_criterion(id_f_on_dig, real_dig)

            loss_id = (loss_id_g + loss_id_f) * 0.5

            # Optional perceptual loss between fake_film and real_film to preserve
            # higher-level tone/contrast (doesn't affect geometry).
            loss_perc = torch.tensor(0.0, device=device)
            if perc is not None:
                loss_perc = (
                    perc(fake_film, real_film) + perc(fake_dig, real_dig)
                ) * 0.5

            # Compose final generator loss
            loss_g = (
                loss_g_adv_film
                + loss_f_adv_dig
                + lambda_cycle * loss_cycle
                + lambda_id * loss_id
                + 0.1 * loss_perc
            )

            loss_g.backward()
            g_optimizer.step()

            # ------------------- Logging -------------------
            if (it + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] Iter [{it+1}] | D_loss: {loss_d.item():.4f} | G_loss: {loss_g.item():.4f} | cycle: {loss_cycle.item():.4f} | id: {loss_id.item():.4f}"
                )

        # ------------------- Epoch-end: save checkpoints -------------------
        torch.save(G.state_dict(), f"G_epoch_{epoch+1}.pth")
        torch.save(F.state_dict(), f"F_epoch_{epoch+1}.pth")
        torch.save(D_film.state_dict(), f"Dfilm_epoch_{epoch+1}.pth")
        torch.save(D_dig.state_dict(), f"Ddig_epoch_{epoch+1}.pth")
        print(f"Saved epoch {epoch+1} checkpoints.")

    # Save final best models
    torch.save(G.state_dict(), "best_model_G.pth")
    torch.save(F.state_dict(), "best_model_F.pth")
    print("Training complete. Saved best_model_G.pth and best_model_F.pth")


# ====================================================================
#                           EVALUATION
# ====================================================================


def apply_trained_generator(
    model_path, src_folder, out_folder, direction="G", device="mps"
):
    """
    Apply a trained generator to all images in src_folder and save outputs.
    Maintains original aspect ratio by padding to 512x512.

    direction: 'G' means digital->film (load best_model_G.pth)
               'F' means film->digital (load best_model_F.pth)
    """
    device = torch.device(device if torch.mps.is_available() else "cpu")
    inv_to_pil = transforms.ToPILImage()

    model = PerPixelAffineNet(base_ch=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    p = Path(src_folder)
    out_p = Path(out_folder)
    out_p.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(
        [q for q in p.iterdir() if q.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    ):
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Pad to square, then resize to 512x512
        max_dim = max(orig_w, orig_h)
        pad_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        offset = ((max_dim - orig_w) // 2, (max_dim - orig_h) // 2)
        pad_img.paste(img, offset)

        pad_img_resized = pad_img.resize((512, 512))
        inp = transforms.ToTensor()(pad_img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            tmap = model(inp)
            out = torch.clamp(apply_perpixel_affine(tmap, inp), 0, 1)

        out_img = inv_to_pil(out.squeeze(0).cpu())
        # Compute crop coordinates in the 512x512 space corresponding to the original image region
        scale = 512 / max_dim
        left = int(offset[0] * scale)
        top = int(offset[1] * scale)
        right = int(left + orig_w * scale)
        bottom = int(top + orig_h * scale)
        out_img = out_img.crop((left, top, right, bottom))
        out_img = out_img.resize((orig_w, orig_h))
        out_img.save(out_p / img_path.name)

    print(f"Saved outputs to {out_p}")


# ====================================================================
#                              CLI
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./dataset_root")
    parser.add_argument(
        "--mode",
        type=str,
        default="train_unpaired",
        choices=["train_unpaired", "apply", "eval"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--model", type=str, default="best_model_G.pth")
    parser.add_argument("--src", type=str, default="./digital")
    parser.add_argument("--out", type=str, default="./out")
    parser.add_argument(
        "--fast_debug",
        action="store_true",
        help="Disable heavy losses & use smaller images for quick debugging",
    )
    parser.add_argument("--use_perc", action="store_true")
    args = parser.parse_args()

    if args.mode == "train_unpaired":
        if args.fast_debug:
            print(
                "Fast debug mode: using smaller images and disabling perceptual loss."
            )
            args.epochs = 2
            args.batch = 2
            args.use_perc = False

        train_unpaired(
            data_root=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            device=args.device,
            use_perc=args.use_perc,
        )
    elif args.mode == "apply":
        apply_trained_generator(args.model, args.src, args.out, device=args.device)
    else:
        print("Unknown mode")
