# import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
print("Using device:", device)


# =========================
# 2) LOAD IMAGES + MASKS
# =========================
images = []
masks = []

found_mask = False

for folder_path in folder_paths:
    for file_path in sorted(glob(folder_path + "/*")):

        img = cv2.imread(file_path)
        img = cv2.resize(img, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0  # normalize

        if "mask" in file_path:
            if found_mask:
                masks[-1] += img
                masks[-1] = np.where(masks[-1] > 0.5, 1.0, 0.0)
            else:
                masks.append(img)
                found_mask = True
        else:
            images.append(img)
            found_mask = False

X = np.array(images, dtype=np.float32)  # [N, H, W]
y = np.array(masks, dtype=np.float32)   # [N, H, W]

# Add channel dimension:
# Keras: [N,H,W,1]
# PyTorch: [N,1,H,W]
X = np.expand_dims(X, axis=1)
y = np.expand_dims(y, axis=1)

print(f"X shape: {X.shape} | y shape: {y.shape}")

# Split data (same as your code)
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
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder
        u1 = self.upconv1(bn)
        cat1 = torch.cat([u1, c4], dim=1)
        c5 = self.conv5(cat1)

        u2 = self.upconv2(c5)
        cat2 = torch.cat([u2, c3], dim=1)
        c6 = self.conv6(cat2)

        u3 = self.upconv3(c6)
        cat3 = torch.cat([u3, c2], dim=1)
        c7 = self.conv7(cat3)

        u4 = self.upconv4(c7)
        cat4 = torch.cat([u4, c1], dim=1)
        c8 = self.conv8(cat4)

        # Output logits (do NOT sigmoid here)
        out = self.out_conv(c8)
        return out


model = UNet().to(device)
print(model)


# =========================
# 5) LOSS + OPTIMIZER
# =========================
# In TF you used sigmoid in model + binary_crossentropy
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
