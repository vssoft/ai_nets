# train_unet_busi.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from unet_busi_loader import load_busi_dataset
from unet_model import UNet


# =========================
# 1) CONFIG
# =========================

# --- Dataset location ---
# Root folder of the BUSI dataset (contains subfolders "benign" and "malignant")
DATA_PATH = "D:/pyton/datasets/Dataset_BUSI_with_GT"

# Explicit paths to the two BUSI classes.
# For segmentation training, we do not use the class label directly,
# but we load all samples from both folders to build the segmentation dataset.
FOLDER_PATHS = [f"{DATA_PATH}/malignant", f"{DATA_PATH}/benign"]

# --- Image preprocessing ---
# All ultrasound images and masks are resized to IMG_SIZE x IMG_SIZE.
# Smaller sizes train faster and use less VRAM but may lose lesion detail.
# Typical values to try: 128, 256.
IMG_SIZE = 128

# Fraction of the dataset used for validation.
# Validation data is not used for training and is used to measure generalization.
# Typical values: 0.1 (10%), 0.2 (20%).
VAL_SPLIT = 0.10

# --- Training hyperparameters ---
# Batch size = number of images processed before each optimizer update.
# Larger batch size trains faster but uses more GPU memory.
# If you see "CUDA out of memory", reduce this first.
BATCH_SIZE = 4

# Number of full passes over the training dataset.
# In segmentation, 30–100 epochs are typical depending on dataset size.
EPOCHS = 40

# Learning rate for Adam optimizer.
# This is the most important hyperparameter for training stability.
# If training becomes unstable or losses become NaN, reduce LR (e.g. 3e-4 or 1e-4).
# If training is extremely slow to improve, you may increase LR slightly.
LR = 1e-3   # Recommended stable alternative: 3e-4

# Threshold used to convert predicted probabilities to binary mask (0/1).
# This affects metrics like Dice and IoU.
# 0.5 is the standard default, but for medical datasets it is sometimes tuned (0.3–0.7).
THRESHOLD = 0.5

# --- Loss function weighting ---
# We train with a combined loss:
#   TotalLoss = BCE_WEIGHT * BCEWithLogitsLoss + DICE_WEIGHT * DiceLoss
#
# BCE focuses on pixel-level correctness; Dice focuses on region overlap.
# If predictions are too conservative (mostly background), increase Dice weight.
# If predictions are too noisy, increase BCE weight.
BCE_WEIGHT = 1.0
DICE_WEIGHT = 1.0

# --- DataLoader performance ---
# Number of subprocesses used to load data in parallel.
# On Windows, setting this too high can cause issues; if you experience crashes/hangs, set to 0.
# On Linux/WSL, values like 4 or 8 are common.
NUM_WORKERS = 2

# When using CUDA, pin_memory=True speeds up host-to-GPU memory transfers.
# It is recommended when training on GPU.
PIN_MEMORY = True

# Automatic Mixed Precision (AMP) uses float16 where possible to speed up training and reduce VRAM.
# AMP can cause instability (NaN losses) especially in segmentation with Dice loss.
# Start with False for stability; enable later if you want performance gains.
USE_AMP = False

# --- Reproducibility ---
# Random seed ensures that:
# - train/validation split is reproducible
# - initialization and shuffling are consistent between runs
# This is essential when comparing experiments.
SEED = 42

# --- Checkpoint saving ---
# Directory where training checkpoints are stored.
# Keeping checkpoints outside Git is best practice (use .gitignore).
CHECKPOINT_DIR = "checkpoints_busi"

# Saved whenever validation IoU improves.
# This is typically the checkpoint you will use later for inference.
BEST_MODEL_NAME = "unet_busi_best.pt"

# Saved at the end of every epoch (overwritten each time).
# Useful if you want to resume training or inspect final model state.
LAST_MODEL_NAME = "unet_busi_last.pt"

# Practical tuning advice
# If you want a quick “good” experimental baseline for BUSI segmentation:
# IMG_SIZE = 256 (if VRAM allows)
# BATCH_SIZE = 2 (if you run out of memory)
# LR = 3e-4
# BCE_WEIGHT = 0.5, DICE_WEIGHT = 1.5 (often improves overlap)
# EPOCHS = 60

# =========================
# 2) REPRODUCIBILITY
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If using GPU:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (slower but reproducible).
    # For maximum speed, you may disable these.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 3) DATASET
# =========================

class SegDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]    # [1,H,W]
        mask = self.y[idx]   # [1,H,W]
        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )


# =========================
# 4) LOSS: BCE + DICE
# =========================

class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation (logits + masks).
    Uses sigmoid(logits) to get probabilities.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)  # [B,1,H,W]
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


def combined_loss_fn(logits, targets, bce_loss, dice_loss):
    bce = bce_loss(logits, targets)
    dice = dice_loss(logits, targets)
    return BCE_WEIGHT * bce + DICE_WEIGHT * dice, bce.item(), dice.item()


# =========================
# 5) METRICS
# =========================

@torch.no_grad()
def compute_metrics(logits, masks, threshold=0.5):
    """
    Returns: pixel_acc, dice_score, iou
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Pixel accuracy
    acc = (preds == masks).float().mean().item()

    # Dice score (binary)
    eps = 1e-7
    intersection = (preds * masks).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
    dice = ((2 * intersection + eps) / (union + eps)).mean().item()

    # IoU (Jaccard) using sklearn - flatten
    preds_np = preds.detach().cpu().numpy().astype(int).ravel()
    masks_np = masks.detach().cpu().numpy().astype(int).ravel()
    iou = jaccard_score(preds_np, masks_np, zero_division=0)

    return acc, dice, iou


# =========================
# 6) TRAIN / VALIDATE
# =========================

def train_one_epoch(model, loader, optimizer, bce_loss, dice_loss, device, scaler=None):
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0

    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(imgs)
                loss, bce_val, dice_val = combined_loss_fn(logits, masks, bce_loss, dice_loss)

            # Stop immediately if NaN
            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during training (AMP).")

            scaler.scale(loss).backward()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(imgs)
            loss, bce_val, dice_val = combined_loss_fn(logits, masks, bce_loss, dice_loss)

            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during training (FP32).")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_bce += bce_val
        total_dice_loss += dice_val

    n = len(loader)
    return total_loss / n, total_bce / n, total_dice_loss / n


@torch.no_grad()
def validate_one_epoch(model, loader, bce_loss, dice_loss, device):
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0

    total_acc = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for imgs, masks in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss, bce_val, dice_val = combined_loss_fn(logits, masks, bce_loss, dice_loss)

        acc, dice, iou = compute_metrics(logits, masks, threshold=THRESHOLD)

        total_loss += loss.item()
        total_bce += bce_val
        total_dice_loss += dice_val

        total_acc += acc
        total_dice += dice
        total_iou += iou

    n = len(loader)
    return (
        total_loss / n,
        total_bce / n,
        total_dice_loss / n,
        total_acc / n,
        total_dice / n,
        total_iou / n,
    )


# =========================
# 7) CHECKPOINTS
# =========================

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


# =========================
# 8) VISUALIZATION
# =========================

@torch.no_grad()
def visualize_one_sample(model, X_val, y_val, device, threshold=0.5):
    model.eval()
    idx = np.random.randint(0, len(X_val))

    img = torch.tensor(X_val[idx:idx+1], dtype=torch.float32).to(device)  # [1,1,H,W]
    gt = y_val[idx, 0]  # [H,W]

    logits = model(img)
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float().cpu().numpy()[0, 0]

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(X_val[idx, 0], cmap="gray")
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# 9) MAIN
# =========================

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nSTART JOB using device:", device)

    # Load dataset
    X, y = load_busi_dataset(FOLDER_PATHS, size=IMG_SIZE, verbose=True)
    print(f"X shape: {X.shape} | y shape: {y.shape}")

    print("X min/max:", X.min(), X.max())
    print("y unique:", np.unique(y))
    print("Any NaN in X?", np.isnan(X).any())
    print("Any NaN in y?", np.isnan(y).any())


    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=SEED
    )

    train_ds = SegDataset(X_train, y_train)
    val_ds = SegDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Model
    model = UNet(in_channels=1, out_channels=1, base_filters=64).to(device)
    print(model)

    # Loss & optimizer
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scaler = torch.amp.GradScaler("cuda") if (USE_AMP and device.type == "cuda") else None
    # scaler = torch.cuda.amp.GradScaler() if (USE_AMP and device.type == "cuda") else None

    # Checkpoint paths
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
    last_path = os.path.join(CHECKPOINT_DIR, LAST_MODEL_NAME)

    best_iou = -1.0

    # Training loop
    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_bce, train_dice = train_one_epoch(
                model, train_loader, optimizer, bce_loss, dice_loss, device, scaler=scaler
            )

            (
                val_loss, val_bce, val_dice_loss,
                val_acc, val_dice, val_iou
            ) = validate_one_epoch(model, val_loader, bce_loss, dice_loss, device)

            # Save last checkpoint every epoch
            save_checkpoint(model, last_path)

            # Save best checkpoint by IoU
            if val_iou > best_iou:
                best_iou = val_iou
                save_checkpoint(model, best_path)
                best_tag = " (BEST)"
            else:
                best_tag = ""

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss={train_loss:.4f} (BCE={train_bce:.4f}, DiceLoss={train_dice:.4f}) | "
                f"Val Loss={val_loss:.4f} (BCE={val_bce:.4f}, DiceLoss={val_dice_loss:.4f}) | "
                f"Val Acc={val_acc:.4f} | Val Dice={val_dice:.4f} | Val IoU={val_iou:.4f}"
                f"{best_tag}"
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")

        # Save a final checkpoint so you don't lose progress
        save_checkpoint(model, last_path)
        print(f"Saved last checkpoint to: {last_path}")

        print("Exiting gracefully.")
        return


    # Visualize one prediction using the best model
    model.load_state_dict(torch.load(best_path, map_location=device))
    visualize_one_sample(model, X_val, y_val, device, threshold=THRESHOLD)


if __name__ == "__main__":
    main()
