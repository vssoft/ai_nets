# unet_busi_pytorch_2.py
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

data_path = "D:/pyton/datasets/Dataset_BUSI_with_GT"
folder_paths = [f"{data_path}/malignant", f"{data_path}/benign"]
size = 128
batch_size = 4
epochs = 40
lr = 1e-3
threshold = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nSTART JOB using device:", device)

# =========================
# 2) LOAD IMAGES + MASKS
# =========================

X, y = load_busi_dataset(folder_paths, size=size, verbose=True)

print(f"X shape: {X.shape} | y shape: {y.shape}")

# print('END\n-------------\n'); sys.exit()  # This will terminate the script here

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# =========================
# 3) DATASET + DATALOADER
# =========================

class SegDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # [1,H,W]
        mask = self.y[idx] # [1,H,W]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


train_ds = SegDataset(X_train, y_train)
val_ds   = SegDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


# =========================
# 4) U-NET MODEL (matching your TF architecture)
# =========================

model = UNet(in_channels=1, out_channels=1, base_filters=64).to(device)
print(model)


# =========================
# 5) LOSS + OPTIMIZER
# =========================
# In PyTorch best practice is:
#   model outputs logits
#   use BCEWithLogitsLoss (includes sigmoid internally)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# =========================
# 6) TRAINING + VALIDATION LOOP
# =========================

def compute_accuracy_and_iou(logits, masks):
    """
    logits: [B,1,H,W] raw model output
    masks:  [B,1,H,W] ground truth (0/1)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Accuracy (pixel-wise)
    correct = (preds == masks).float().mean().item()

    # IoU (Jaccard)
    # Flatten
    preds_np = preds.detach().cpu().numpy().astype(int).ravel()
    masks_np = masks.detach().cpu().numpy().astype(int).ravel()

    iou = jaccard_score(preds_np, masks_np)

    return correct, iou


for epoch in range(epochs):
    model.train()
    train_loss = 0

    for imgs, masks_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        imgs = imgs.to(device)
        masks_batch = masks_batch.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_acc = 0
    val_iou = 0

    with torch.no_grad():
        for imgs, masks_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            imgs = imgs.to(device)
            masks_batch = masks_batch.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks_batch)
            val_loss += loss.item()

            acc, iou = compute_accuracy_and_iou(logits, masks_batch)
            val_acc += acc
            val_iou += iou

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_iou /= len(val_loader)

    print(f"Epoch {epoch+1:02d}: "
          f"Train Loss={train_loss:.4f} | "
          f"Val Loss={val_loss:.4f} | "
          f"Val Acc={val_acc:.4f} | "
          f"Val IoU={val_iou:.4f}")


# =========================
# 7) FINAL IOU ON WHOLE VAL SET (same as your TF code)
# =========================

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for imgs, masks_batch in val_loader:
        imgs = imgs.to(device)
        masks_batch = masks_batch.to(device)

        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).int()

        all_preds.append(preds.cpu().numpy().ravel())
        all_true.append(masks_batch.cpu().numpy().astype(int).ravel())

all_preds = np.concatenate(all_preds)
all_true = np.concatenate(all_true)

iou = jaccard_score(all_preds, all_true)
print(f" IoU (Jaccard Score): {iou:.4f}")


# =========================
# 8) VISUALIZE ONE EXAMPLE (same as your last plot)
# =========================

i = 6  # choose any index within validation set length

img = torch.tensor(X_val[i:i+1], dtype=torch.float32).to(device)   # [1,1,H,W]
gt  = y_val[i, 0]  # [H,W]

model.eval()
with torch.no_grad():
    logits = model(img)
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float().cpu().numpy()[0, 0]

plt.subplot(1, 3, 1)
plt.imshow(X_val[i, 0], cmap="gray")
plt.title("Input")

plt.subplot(1, 3, 2)
plt.imshow(gt, cmap="gray")
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(pred, cmap="gray")
plt.title("Prediction")

plt.show()
