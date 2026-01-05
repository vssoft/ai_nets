import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from unet_busi_loader import load_busi_dataset
from unet_model import UNet

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "D:/pyton/datasets/Dataset_BUSI_with_GT"
FOLDER_PATHS = [f"{DATA_PATH}/malignant", f"{DATA_PATH}/benign"]
IMG_SIZE = 128
THRESHOLD = 0.5

CHECKPOINT_PATH = "checkpoints_busi/unet_busi_best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Load dataset
# -------------------------
X, y = load_busi_dataset(FOLDER_PATHS, size=IMG_SIZE, verbose=False)
print("Loaded dataset:", X.shape, y.shape)

# -------------------------
# Load trained model
# -------------------------
model = UNet(in_channels=1, out_channels=1, base_filters=64).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print("Loaded model:", CHECKPOINT_PATH)

# -------------------------
# Function to visualize one image
# -------------------------
@torch.no_grad()
def show_sample(idx):
    img = torch.tensor(X[idx:idx+1], dtype=torch.float32).to(device)  # [1,1,H,W]
    gt = y[idx, 0]  # [H,W]

    logits = model(img)
    probs = torch.sigmoid(logits)
    pred = (probs > THRESHOLD).float().cpu().numpy()[0, 0]

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(X[idx, 0], cmap="gray")
    plt.title(f"Input (idx={idx})")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# -------------------------
# Inspect multiple images
# -------------------------
# Example: show a few random images
# for idx in np.random.choice(len(X), size=5, replace=False):
#     show_sample(int(idx))

# show_sample(0)

# while True:
#     s = input("Enter index (or q): ")
#     if s.lower() == "q":
#         break
#     show_sample(int(s))

# for idx in range(1, 13):
#     show_sample(idx)
#     time.sleep(1.5)   # wait 1.5 seconds

@torch.no_grad()
def show_range(start=1, end=12):
    model.eval()

    n = end - start + 1
    plt.figure(figsize=(10, 3 * n))

    row = 0
    for idx in range(start, end + 1):
        img = torch.tensor(X[idx:idx+1], dtype=torch.float32).to(device)
        gt = y[idx, 0]

        logits = model(img)
        probs = torch.sigmoid(logits)
        pred = (probs > THRESHOLD).float().cpu().numpy()[0, 0]

        # Input
        plt.subplot(n, 3, row*3 + 1)
        plt.imshow(X[idx, 0], cmap="gray")
        plt.title(f"Input idx={idx}")
        plt.axis("off")

        # Ground truth
        plt.subplot(n, 3, row*3 + 2)
        plt.imshow(gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        # Prediction
        plt.subplot(n, 3, row*3 + 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        row += 1

    plt.tight_layout()
    plt.show()


# Call it:
# show_range(1, 12)

@torch.no_grad()
def show_sample2(idx, delay=1.5):
    img = torch.tensor(X[idx:idx+1], dtype=torch.float32).to(device)
    gt = y[idx, 0]

    logits = model(img)
    probs = torch.sigmoid(logits)
    pred = (probs > THRESHOLD).float().cpu().numpy()[0, 0]

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(X[idx, 0], cmap="gray")
    plt.title(f"Input idx={idx}")
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

    # --- Non-blocking display ---
    plt.show(block=False)     # do not block script execution
    plt.pause(delay)          # keep figure visible for `delay` seconds
    plt.close()               # close the figure automatically

# Call it:
for idx in range(1, 13):
    # show_sample2(idx, delay=2.0)
    show_sample(idx)
