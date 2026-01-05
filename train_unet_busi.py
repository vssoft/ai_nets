# train_unet_busi.py
# BUSI (Breast Ultrasound Images) Dataset segmentation training script.

import os
import random
import time
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
    """
    Set random seeds to make results reproducible.

    Why this matters:
    - Deep learning training involves randomness:
        * train/val splitting
        * shuffling batches
        * weight initialization
        * dropout (if used)
        * some GPU kernels may be nondeterministic
    - Without fixed seeds, two runs with identical code may produce different results,
      which makes experimentation and debugging harder.

    Note:
    - Even with fixed seeds, complete determinism is not always guaranteed on GPU
      due to certain CUDA operations, but this gets very close.
    """

    # Python's built-in random module (affects any random.* usage)
    random.seed(seed)

    # NumPy random generator (affects np.random usage: data shuffling, sampling, etc.)
    np.random.seed(seed)

    # PyTorch CPU random generator (affects initialization, CPU ops, etc.)
    torch.manual_seed(seed)

    # If using GPU, also fix CUDA random generator(s).
    # manual_seed_all sets the seed for all available CUDA devices (GPU 0, GPU 1, etc.)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN is NVIDIA's deep neural network library used by PyTorch for many ops.
    # These flags control determinism vs performance:
    #
    # deterministic=True:
    # - Forces cuDNN to use deterministic algorithms where possible.
    # - Makes results more reproducible.
    # - Can be slower.
    torch.backends.cudnn.deterministic = True

    # benchmark=False:
    # - Disables cuDNN auto-tuning.
    # - If benchmark=True, cuDNN tries multiple convolution algorithms at startup
    #   and picks the fastest for your input shapes, which is faster overall.
    # - However, benchmark=True can make runs slightly nondeterministic.
    torch.backends.cudnn.benchmark = False

# =========================
# 3) DATASET
# =========================

class SegDataset(Dataset):
    """
    PyTorch Dataset wrapper for segmentation data.

    Purpose:
    - Provides a standard interface so PyTorch's DataLoader can:
        * iterate over the dataset sample-by-sample
        * shuffle samples
        * create batches
        * load data in parallel (num_workers > 0)
    - Each sample is a tuple: (image_tensor, mask_tensor)

    Input format:
    - X is a NumPy array of shape [N, 1, H, W] (grayscale images)
    - y is a NumPy array of shape [N, 1, H, W] (binary masks)

    Output:
    - returns image and mask as torch.float32 tensors
    - float32 is required for:
        * convolution layers
        * BCEWithLogitsLoss / DiceLoss
    """

    def __init__(self, X, y):
        # Store references to the full image and mask arrays.
        # We do not copy the data here; we just keep pointers.
        self.X = X
        self.y = y

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Required by PyTorch so it knows how many indices exist.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns one sample (image, mask) at index idx.
        PyTorch DataLoader calls this repeatedly to build batches.

        img shape:  [1, H, W]  (single channel grayscale)
        mask shape: [1, H, W]  (binary mask 0/1)

        Note:
        - We convert NumPy arrays to torch tensors here.
        - This keeps the DataLoader interface consistent.
        - Later, this is also the place where you typically apply augmentations.
        """
        img = self.X[idx]    # [1, H, W]
        mask = self.y[idx]   # [1, H, W]

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

# =========================
# 4) LOSS: BCE + DICE
# =========================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Background:
    - Dice score measures overlap between predicted mask and ground truth mask:
        Dice = (2 * |Prediction ∩ GroundTruth|) / (|Prediction| + |GroundTruth|)
    - Dice Loss is defined as:
        DiceLoss = 1 - DiceScore
    - Dice is widely used in medical imaging segmentation because lesions often occupy
      a small fraction of the image (strong class imbalance). BCE alone may lead to
      predicting mostly background, which still gives high pixel accuracy.

    Inputs:
    - logits: raw model outputs of shape [B, 1, H, W] (NOT sigmoid probabilities)
    - targets: ground truth masks of shape [B, 1, H, W] (values should be 0 or 1)

    Important:
    - We apply sigmoid(logits) inside this loss to convert logits → probabilities.
    - We flatten each image into a vector so that Dice is computed per sample.
    """

    def __init__(self, eps=1e-7):
        """
        eps is a small constant that prevents division by zero when the union is 0.
        This can happen if both prediction and target are all zeros.
        """
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        # Convert logits → probabilities in [0,1]
        probs = torch.sigmoid(logits)  # [B,1,H,W]

        # Flatten each sample so Dice is computed per image (per batch element)
        probs = probs.view(probs.size(0), -1)       # [B, H*W]
        targets = targets.view(targets.size(0), -1) # [B, H*W]

        # Intersection and union (per sample)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        # Dice coefficient per sample, then average across batch
        dice = (2 * intersection + self.eps) / (union + self.eps)

        # Return Dice loss
        return 1 - dice.mean()


def combined_loss_fn(logits, targets, bce_loss, dice_loss):
    """
    Combined loss = weighted BCE + weighted Dice loss.

    Why combine them?
    - BCEWithLogitsLoss gives stable pixel-wise gradients (local supervision).
    - DiceLoss optimizes region-level overlap (global supervision).
    - The combination usually performs better than either alone for segmentation.

    Returns:
    - total_loss: scalar tensor used for backprop
    - bce_value: float for logging
    - dice_value: float for logging
    """
    bce = bce_loss(logits, targets)      # BCE computed from logits (no sigmoid needed)
    dice = dice_loss(logits, targets)    # Dice computed from sigmoid(logits)

    total = BCE_WEIGHT * bce + DICE_WEIGHT * dice
    return total, bce.item(), dice.item()

# =========================
# 5) METRICS
# =========================

@torch.no_grad()
def compute_metrics(logits, masks, threshold=0.5):
    """
    Compute evaluation metrics for binary segmentation.

    Inputs:
    - logits: raw model outputs of shape [B, 1, H, W]
              (these are NOT probabilities; they are unbounded real values)
    - masks:  ground truth binary masks of shape [B, 1, H, W]
              (values should be 0 or 1)
    - threshold: decision threshold for converting probabilities → binary predictions.
                 Default 0.5 is standard, but it can be tuned.

    Returns:
    - acc:  pixel-wise accuracy (fraction of pixels correctly classified)
    - dice: Dice score (overlap measure; good for medical segmentation)
    - iou:  Intersection-over-Union (Jaccard index; stricter overlap measure)

    Note:
    - We use @torch.no_grad() because metrics are computed only for evaluation
      and do not need gradients. This saves GPU memory and speeds computation.
    """

    # Convert logits → probabilities (0..1)
    probs = torch.sigmoid(logits)

    # Convert probabilities → hard binary mask using threshold
    preds = (probs > threshold).float()

    # -------------------------
    # 1) Pixel Accuracy
    # -------------------------
    # Accuracy counts correct pixels / total pixels.
    # WARNING: In segmentation tasks with large background regions (common in medical images),
    # accuracy can be misleadingly high even if the model fails to segment the lesion.
    # Example: If 90% of pixels are background, predicting all zeros gives 90% accuracy.
    acc = (preds == masks).float().mean().item()

    # -------------------------
    # 2) Dice Score
    # -------------------------
    # Dice score measures overlap between prediction and ground truth.
    # Dice = 2*|A ∩ B| / (|A| + |B|)
    # Values:
    # - 1.0 = perfect overlap
    # - 0.0 = no overlap
    eps = 1e-7  # prevents division by zero

    intersection = (preds * masks).sum(dim=(1, 2, 3))  # per-sample overlap
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))  # per-sample total area

    dice = ((2 * intersection + eps) / (union + eps)).mean().item()

    # -------------------------
    # 3) IoU (Jaccard Index)
    # -------------------------
    # IoU = |A ∩ B| / |A ∪ B|
    # It is a stricter overlap measure than Dice.
    #
    # We use sklearn.metrics.jaccard_score here, which expects 1D arrays.
    # So we flatten all pixels across the batch.
    preds_np = preds.detach().cpu().numpy().astype(int).ravel()
    masks_np = masks.detach().cpu().numpy().astype(int).ravel()

    # zero_division=0 prevents errors when both prediction and mask contain no positives
    iou = jaccard_score(preds_np, masks_np, zero_division=0)

    return acc, dice, iou

# =========================
# 6) TRAIN / VALIDATE
# =========================

def train_one_epoch(model, loader, optimizer, bce_loss, dice_loss, device, scaler=None):
    """
    Train the model for one full epoch (one pass through the training dataset).

    Args:
    - model: the U-Net model (nn.Module)
    - loader: DataLoader providing training batches (imgs, masks)
    - optimizer: optimizer instance (e.g., Adam)
    - bce_loss: BCEWithLogitsLoss instance
    - dice_loss: DiceLoss instance
    - device: torch.device('cuda') or torch.device('cpu')
    - scaler: GradScaler for AMP training; if None, runs in normal float32 mode

    Returns:
    - mean_total_loss: average combined loss over the epoch
    - mean_bce_loss: average BCE component over the epoch
    - mean_dice_loss: average DiceLoss component over the epoch
    """

    # Set the model to training mode:
    # - enables Dropout (if present)
    # - enables BatchNorm updates (if present)
    # U-Net here does not use them by default, but train() is still required.
    model.train()

    # Accumulators for logging epoch-level average loss values
    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0

    # Loop over batches:
    # imgs:  [B, 1, H, W]
    # masks: [B, 1, H, W]
    for imgs, masks in tqdm(loader, desc="Train", leave=False):

        # Move tensors to GPU/CPU.
        # non_blocking=True can speed up transfers when DataLoader uses pinned memory.
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Clear gradients from the previous iteration.
        # set_to_none=True is slightly faster and can reduce memory overhead.
        optimizer.zero_grad(set_to_none=True)

        # ---------------------------
        # AMP (Automatic Mixed Precision) branch
        # ---------------------------
        # If scaler is provided, training runs with mixed precision (float16 where safe),
        # which can accelerate training and reduce VRAM usage.
        # However, AMP can cause numerical instability in segmentation, so we handle NaNs.
        if scaler is not None:
            # autocast runs forward pass in mixed precision
            with torch.amp.autocast(device_type="cuda"):
                logits = model(imgs)  # raw outputs (logits)
                loss, bce_val, dice_val = combined_loss_fn(logits, masks, bce_loss, dice_loss)

            # Stop immediately if NaN is detected.
            # If a NaN reaches the optimizer step, weights become corrupted permanently.
            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during training (AMP).")

            # Scale loss to reduce underflow in float16 gradients
            scaler.scale(loss).backward()

            # Before clipping gradients, we must "unscale" them back to real scale.
            # Otherwise clipping would be applied to scaled gradients and be incorrect.
            scaler.unscale_(optimizer)

            # Gradient clipping prevents exploding gradients, a common cause of NaNs.
            # max_norm=1.0 is a stable default for segmentation training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Apply optimizer step with scaled gradients
            scaler.step(optimizer)

            # Update the scaling factor dynamically
            scaler.update()

        # ---------------------------
        # Standard float32 training branch
        # ---------------------------
        else:
            logits = model(imgs)
            loss, bce_val, dice_val = combined_loss_fn(logits, masks, bce_loss, dice_loss)

            # NaN check (should rarely happen in FP32, but still good practice)
            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during training (FP32).")

            # Backpropagate loss -> compute gradients
            loss.backward()

            # Clip gradients for stability (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

        # Update running totals for epoch-level statistics
        total_loss += loss.item()
        total_bce += bce_val
        total_dice_loss += dice_val

    # Average across all training batches in this epoch
    n = len(loader)
    return total_loss / n, total_bce / n, total_dice_loss / n

@torch.no_grad()
def validate_one_epoch(model, loader, bce_loss, dice_loss, device):
    """
    Run one full validation epoch (one pass through the validation dataset).

    Validation is used to measure how well the model generalizes to unseen data.
    Unlike training:
    - We do NOT compute gradients
    - We do NOT update weights
    - We run with model.eval() so layers behave correctly

    Args:
    - model: the trained model (nn.Module)
    - loader: DataLoader providing validation batches
    - bce_loss: BCEWithLogitsLoss instance
    - dice_loss: DiceLoss instance
    - device: torch.device('cuda') or torch.device('cpu')

    Returns (averaged over validation batches):
    - val_total_loss: combined loss (BCE + DiceLoss)
    - val_bce_loss: BCE component only
    - val_dice_loss: DiceLoss component only
    - val_acc: pixel-wise accuracy (may be misleading when background dominates)
    - val_dice: Dice score (overlap metric; most important for medical segmentation)
    - val_iou: IoU score (Jaccard index; stricter overlap metric)
    """

    # Disable training-specific behavior (Dropout, BatchNorm updates, etc.)
    # This ensures consistent evaluation results.
    model.eval()

    # Accumulators for losses
    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0

    # Accumulators for metrics
    total_acc = 0.0
    total_dice = 0.0
    total_iou = 0.0

    # Loop over validation batches
    for imgs, masks in tqdm(loader, desc="Val", leave=False):

        # Move to GPU/CPU. non_blocking=True helps when pin_memory=True is used.
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward pass only (no gradients due to @torch.no_grad())
        logits = model(imgs)

        # Compute validation loss (same combined loss used in training)
        loss, bce_val, dice_val = combined_loss_fn(logits, masks, bce_loss, dice_loss)

        # Compute evaluation metrics using hard-thresholded predictions
        # Note: This is for evaluation only (not differentiable), so it is safe here.
        acc, dice, iou = compute_metrics(logits, masks, threshold=THRESHOLD)

        # Accumulate batch-level values for epoch-level averages
        total_loss += loss.item()
        total_bce += bce_val
        total_dice_loss += dice_val

        total_acc += acc
        total_dice += dice
        total_iou += iou

    # Convert totals to means by dividing by number of batches
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
    """
    Save model weights to disk.

    Why checkpoints matter:
    - Training can be interrupted (Ctrl+C, crash, power loss).
    - The model can overfit after reaching its best performance.
    - You often want to keep:
        * the best model (highest validation IoU/Dice)
        * the last model (for resuming training)

    What is saved here:
    - model.state_dict(): a dictionary of all learnable parameters (weights/biases)
      and buffers needed to restore the model.

    Note:
    - This function saves ONLY the model weights.
    - If you later want to resume training exactly (including optimizer momentum),
      you should save optimizer state and epoch as well (see optional version below).
    """

    # Ensure the checkpoint directory exists
    # Example: "checkpoints_busi/unet_busi_best.pt"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the model parameters to the given path
    torch.save(model.state_dict(), path)

def format_time(seconds: float) -> str:
    """
    Format seconds as hh:mm:ss for nice printing.
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# =========================
# 8) VISUALIZATION
# =========================

@torch.no_grad()
def visualize_one_sample(model, X_val, y_val, device, threshold=0.5):
    """
    Visualize model prediction on one random validation sample.

    Purpose:
    - Provides a quick qualitative sanity check during/after training.
    - Metrics (Dice/IoU) tell you "how good" numerically,
      but visualization lets you see *what the model is actually predicting*.
    - Useful for debugging: you can detect issues like:
        * model predicts only background
        * model predicts noisy masks
        * mask alignment problems
        * poor lesion boundary accuracy

    Inputs:
    - model: trained segmentation model
    - X_val: validation images as NumPy array [N, 1, H, W]
    - y_val: validation masks as NumPy array [N, 1, H, W]
    - device: torch.device('cuda') or torch.device('cpu')
    - threshold: probability threshold used to convert predicted probabilities → binary mask
    """

    # Switch model to evaluation mode:
    # ensures consistent behavior (disables dropout, uses fixed batchnorm stats, etc.)
    model.eval()

    # Pick a random validation sample index
    idx = np.random.randint(0, len(X_val))

    # Create a batch with one image:
    # X_val[idx:idx+1] keeps the batch dimension (shape becomes [1, 1, H, W])
    img = torch.tensor(X_val[idx:idx+1], dtype=torch.float32).to(device)

    # Extract corresponding ground truth mask (remove channel dimension for plotting)
    gt = y_val[idx, 0]  # [H, W]

    # Forward pass: model outputs logits (raw unbounded values)
    logits = model(img)

    # Convert logits → probabilities in [0,1]
    probs = torch.sigmoid(logits)

    # Convert probabilities → binary mask using threshold
    # pred will be shape [1,1,H,W] so we index [0,0] to get [H,W] for plotting
    pred = (probs > threshold).float().cpu().numpy()[0, 0]

    # Plot input image, ground truth mask, and predicted mask side-by-side
    plt.figure(figsize=(9, 3))

    # Input image (grayscale ultrasound)
    plt.subplot(1, 3, 1)
    plt.imshow(X_val[idx, 0], cmap="gray")
    plt.title("Input")
    plt.axis("off")

    # Ground truth (manual annotation)
    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    # Prediction (model output)
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    # Additionally, show the predicted probability map
    prob_map = probs.cpu().numpy()[0, 0]
    plt.subplot(1, 4, 4)
    plt.imshow(prob_map, cmap="gray")
    plt.title("Probabilities")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# 9) MAIN
# =========================

def main():
    """
    Main training entry point.

    High-level flow:
    1) Set random seed for reproducibility
    2) Select device (GPU if available)
    3) Load and validate dataset
    4) Split into train/validation subsets
    5) Build DataLoaders
    6) Initialize model, loss functions, optimizer (and AMP scaler if enabled)
    7) Train for EPOCHS, evaluate on validation, save checkpoints
    8) Load best model and visualize one prediction sample
    """

    # Ensure consistent results between runs (same split, init, shuffling, etc.)
    set_seed(SEED)

    # Select training device.
    # If CUDA is available, training will run on the GPU (much faster).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nSTART JOB using device:", device)

    # -------------------------
    # Load dataset (preprocessed to NumPy arrays)
    # -------------------------
    # Returns:
    #   X: [N, 1, IMG_SIZE, IMG_SIZE] float32 images normalized to [0,1]
    #   y: [N, 1, IMG_SIZE, IMG_SIZE] float32 binary masks (0 or 1)
    X, y = load_busi_dataset(FOLDER_PATHS, size=IMG_SIZE, verbose=True)
    print(f"X shape: {X.shape} | y shape: {y.shape}")

    # -------------------------
    # Sanity checks (very useful for debugging)
    # -------------------------
    # Confirm that:
    # - images are normalized correctly
    # - masks are binary
    # - there are no NaNs (NaNs can cause training to collapse)
    print("X min/max:", X.min(), X.max())
    print("y unique:", np.unique(y))
    print("Any NaN in X?", np.isnan(X).any())
    print("Any NaN in y?", np.isnan(y).any())

    # -------------------------
    # Train / Validation split
    # -------------------------
    # VAL_SPLIT fraction goes to validation. Training uses the remaining data.
    # random_state=SEED ensures the split is reproducible.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=SEED
    )

    # Wrap NumPy arrays into Dataset objects.
    # Dataset defines __len__ and __getitem__ so DataLoader can batch and shuffle.
    train_ds = SegDataset(X_train, y_train)
    val_ds = SegDataset(X_val, y_val)

    # Create DataLoaders for efficient batching.
    # num_workers > 0 enables parallel loading (can speed up training).
    # pin_memory=True speeds up CPU → GPU transfer when using CUDA.
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # -------------------------
    # Model initialization
    # -------------------------
    # U-Net outputs logits (raw values). Sigmoid is applied later in:
    # - Dice loss
    # - metric computation
    # - visualization
    model = UNet(in_channels=1, out_channels=1, base_filters=64).to(device)
    print(model)

    # -------------------------
    # Loss functions and optimizer
    # -------------------------
    # BCEWithLogitsLoss is stable for binary segmentation and expects logits directly.
    # DiceLoss improves overlap learning and handles class imbalance better.
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    # Adam is a robust default optimizer for segmentation.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -------------------------
    # AMP scaler (optional)
    # -------------------------
    # AMP can speed up training and reduce VRAM usage by using float16 operations,
    # but it may cause numerical instability in segmentation (NaNs).
    # scaler is enabled only if:
    # - USE_AMP=True
    # - device is CUDA
    scaler = torch.amp.GradScaler("cuda") if (USE_AMP and device.type == "cuda") else None

    # -------------------------
    # Checkpoint paths
    # -------------------------
    # We save:
    # - last_path: always overwritten each epoch (latest checkpoint)
    # - best_path: updated when validation IoU improves (best-performing model)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
    last_path = os.path.join(CHECKPOINT_DIR, LAST_MODEL_NAME)

    # Track the best IoU achieved so far (start below any possible IoU)
    best_iou = -1.0

    # -------------------------
    # Training loop
    # -------------------------
    training_start_time = time.time()

    # We wrap the loop in try/except to allow Ctrl+C stopping without losing progress.
    try:
        for epoch in range(1, EPOCHS + 1):
            # Start timer for this epoch
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss, train_bce, train_dice = train_one_epoch(
                model, train_loader, optimizer, bce_loss, dice_loss, device, scaler=scaler
            )

            # Validate for one epoch (no weight updates)
            (
                val_loss, val_bce, val_dice_loss,
                val_acc, val_dice, val_iou
            ) = validate_one_epoch(model, val_loader, bce_loss, dice_loss, device)

            # Save latest checkpoint every epoch (helps resume training after interruption)
            save_checkpoint(model, last_path)

            # Save best checkpoint (selected by IoU)
            # IoU is a strong metric for segmentation overlap quality.
            if val_iou > best_iou:
                best_iou = val_iou
                save_checkpoint(model, best_path)
                best_tag = " (BEST)"
            else:
                best_tag = ""

            # --- Timing ---
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - training_start_time

            avg_epoch_time = total_time / epoch
            eta_time = avg_epoch_time * (EPOCHS - epoch)

            # --- Print epoch summary ---
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss={train_loss:.4f} (BCE={train_bce:.4f}, DiceLoss={train_dice:.4f}) | "
                f"Val Loss={val_loss:.4f} (BCE={val_bce:.4f}, DiceLoss={val_dice_loss:.4f}) | "
                f"Val Acc={val_acc:.4f} | Val Dice={val_dice:.4f} | Val IoU={val_iou:.4f}"
                f"{best_tag} | "
                f"Epoch Time={format_time(epoch_time)} | "
                f"Total={format_time(total_time)} | "
                f"ETA={format_time(eta_time)}"
            )

    except KeyboardInterrupt:
        # User requested stop via Ctrl+C
        print("\nTraining interrupted by user (Ctrl+C).")

        # Save a final checkpoint so you don't lose progress
        save_checkpoint(model, last_path)
        print(f"Saved last checkpoint to: {last_path}")

        print("Exiting gracefully.")
        return

    # -------------------------
    # Post-training visualization
    # -------------------------
    # Reload best model weights before visualization.
    # This ensures you visualize the best-performing checkpoint, not just the last epoch.
    model.load_state_dict(torch.load(best_path, map_location=device))
    visualize_one_sample(model, X_val, y_val, device, threshold=THRESHOLD)


# Standard Python entry point check:
# ensures main() runs only when this script is executed directly.
if __name__ == "__main__":
    main()

# End of train_unet_busi.py
