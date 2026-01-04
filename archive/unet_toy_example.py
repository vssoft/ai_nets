# UNet Toy Example in PyTorch
# This script defines a small U-Net model, creates a toy dataset, and trains the model.
# It then visualizes the results on a sample image.
## Requirements:
# - torch
# - matplotlib
# - numpy
# To run: python unet_toy_example.py in a CPU or GPU environment using (dl-gpu)


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# -------------------------
# 1. Define a small U-Net
# -------------------------

class DoubleConv(nn.Module):
    """
    Two Conv2d + ReLU blocks as in classic U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    """
    A simplified U-Net:
    - Input: 1 x 64 x 64
    - Two downsampling steps, bottleneck, two upsampling steps
    - Output: 1 x 64 x 64 (logits for binary mask)
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(32, 64)

        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)  # 32 from up2 + 32 from skip

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(32, 16)  # 16 from up1 + 16 from skip

        # Output layer (1 channel = binary segmentation)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)          # [B,16,64,64]
        p1 = self.pool1(d1)         # [B,16,32,32]

        d2 = self.down2(p1)         # [B,32,32,32]
        p2 = self.pool2(d2)         # [B,32,16,16]

        # Bottleneck
        b = self.bottleneck(p2)     # [B,64,16,16]

        # Decoder
        u2 = self.up2(b)            # [B,32,32,32]
        cat2 = torch.cat([u2, d2], dim=1)  # [B,64,32,32]
        c2 = self.conv_up2(cat2)    # [B,32,32,32]

        u1 = self.up1(c2)           # [B,16,64,64]
        cat1 = torch.cat([u1, d1], dim=1)  # [B,32,64,64]
        c1 = self.conv_up1(cat1)    # [B,16,64,64]

        out = self.out_conv(c1)     # [B,1,64,64]
        return out


# -------------------------
# 2. Create a toy dataset
# -------------------------

def create_toy_dataset(num_samples=200, img_size=64):
    """
    Toy data: random grayscale images (0..1).
    Target mask: pixels > 0.5 are foreground (1), others background (0).
    This is not meaningful segmentation, but enough to test the pipeline.
    """
    # Images
    x = torch.rand(num_samples, 1, img_size, img_size)

    # Masks: threshold at 0.5
    y = (x > 0.5).float()

    return x, y


# -------------------------
# 3. Training loop
# -------------------------

def train_unet():
    # device = torch.device("cpu")  # CPU-only for now
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available VSS
    print("Using device:", device) # VSS


    # Create model, loss, optimizer
    model = SmallUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create toy data
    x, y = create_toy_dataset(num_samples=200, img_size=64)
    x, y = x.to(device), y.to(device)

    # Simple train loop
    num_epochs = 5
    batch_size = 8

    print("Starting training on CPU...")

    for epoch in range(num_epochs):
        permutation = torch.randperm(x.size(0))
        epoch_loss = 0.0

        for i in range(0, x.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            # batch_x, batch_y = x[indices], y[indices]
            batch_x, batch_y = x[indices].to(device), y[indices].to(device) # VSS
            # model = SmallUNet().to(device) 

            optimizer.zero_grad()
            outputs = model(batch_x)          # [B,1,64,64]

            # if device.type == "cuda" and epoch == 0 and i == 0:
            #     print(torch.cuda.memory_allocated() / 1e6, "MB allocated (after forward)")
            if device.type == "cuda" and epoch == 0 and i == 0:
                    print("Allocated:", torch.cuda.memory_allocated() / 1e6, "MB",
                        "| Reserved:", torch.cuda.memory_reserved() / 1e6, "MB")
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (x.size(0) / batch_size)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    print("Training finished.")
    return model, x, y


# -------------------------
# 4. Visualization
# -------------------------

def visualize_example(model, x, y, index=0):
    model.eval()
    with torch.no_grad():
        img = x[index:index+1]  # [1,1,64,64]
        gt = y[index:index+1]

        logits = model(img)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float()

    # Convert to numpy for plotting
    img_np = img.squeeze().cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()
    pred_np = pred_mask.squeeze().cpu().numpy()

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.title("Input image")
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground truth mask")
    plt.imshow(gt_np, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted mask")
    plt.imshow(pred_np, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# 5. Main
# -------------------------

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    model, x, y = train_unet()
    visualize_example(model, x, y, index=0)
