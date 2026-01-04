import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# -------------------------
# 1. Define a small U-Net
# -------------------------

class DoubleConv(nn.Module):
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
        self.conv_up2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(32, 16)

        # Output layer
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)          # [B,16,64,64]
        p1 = self.pool1(d1)         # [B,16,32,32]

        d2 = self.down2(p1)         # [B,32,32,32]
        p2 = self.pool2(d2)         # [B,32,16,16]

        b = self.bottleneck(p2)     # [B,64,16,16]

        u2 = self.up2(b)            # [B,32,32,32]
        cat2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv_up2(cat2)

        u1 = self.up1(c2)           # [B,16,64,64]
        cat1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv_up1(cat1)

        out = self.out_conv(c1)     # [B,1,64,64]
        return out


# -------------------------
# 2. Synthetic shape dataset
# -------------------------

def generate_shape_sample(img_size=64):
    """
    Generates:
    - grayscale image: shape + noise
    - binary mask: shape region
    Returns (image, mask) as NumPy arrays
    """

    # Start with blank image and blank mask
    img = np.zeros((img_size, img_size), dtype=np.float32)
    mask = np.zeros((img_size, img_size), dtype=np.float32)

    # Randomly choose shape type
    shape_type = np.random.choice(["circle", "rectangle"])

    if shape_type == "circle":
        # Random radius and center
        radius = np.random.randint(img_size // 8, img_size // 4)
        cx = np.random.randint(radius, img_size - radius)
        cy = np.random.randint(radius, img_size - radius)

        # Create circle using distance formula
        Y, X = np.ogrid[:img_size, :img_size]
        dist = (X - cx) ** 2 + (Y - cy) ** 2
        circle = dist <= radius ** 2

        mask[circle] = 1.0
        img[circle] = np.random.uniform(0.6, 1.0)  # bright object

    else:  # rectangle
        w = np.random.randint(img_size // 8, img_size // 3)
        h = np.random.randint(img_size // 8, img_size // 3)
        x1 = np.random.randint(0, img_size - w)
        y1 = np.random.randint(0, img_size - h)

        mask[y1:y1+h, x1:x1+w] = 1.0
        img[y1:y1+h, x1:x1+w] = np.random.uniform(0.6, 1.0)

    # Add Gaussian noise
    noise = np.random.normal(0, 0.10, size=(img_size, img_size)).astype(np.float32)
    img = img + noise

    # Clamp to [0,1]
    img = np.clip(img, 0.0, 1.0)

    return img, mask


def create_shape_dataset(num_samples=500, img_size=64):
    """
    Create a dataset of shape images and masks.
    Returns torch tensors:
      x: [N,1,H,W]
      y: [N,1,H,W]
    """
    images = []
    masks = []

    for _ in range(num_samples):
        img, mask = generate_shape_sample(img_size)
        images.append(img)
        masks.append(mask)

    x = torch.tensor(np.array(images)).unsqueeze(1)  # [N,1,H,W]
    y = torch.tensor(np.array(masks)).unsqueeze(1)   # [N,1,H,W]

    return x, y


# -------------------------
# 3. Training
# -------------------------

def train_unet_on_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = SmallUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Dataset
    x, y = create_shape_dataset(num_samples=800, img_size=64)
    x, y = x.to(device), y.to(device)

    num_epochs = 8
    batch_size = 16

    for epoch in range(num_epochs):
        permutation = torch.randperm(x.size(0))
        epoch_loss = 0.0

        for i in range(0, x.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x[indices], y[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (x.size(0) / batch_size)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    return model, x, y


# -------------------------
# 4. Visualization
# -------------------------

def visualize_predictions(model, x, y, num_examples=3):
    model.eval()

    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()

    with torch.no_grad():
        for i in range(num_examples):
            idx = np.random.randint(0, x_cpu.shape[0])
            img = x_cpu[idx:idx+1]
            gt = y_cpu[idx:idx+1]

            logits = model(img.to(next(model.parameters()).device))
            probs = torch.sigmoid(logits).cpu()
            pred = (probs > 0.5).float()

            img_np = img.squeeze().numpy()
            gt_np = gt.squeeze().numpy()
            pred_np = pred.squeeze().numpy()

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
    model, x, y = train_unet_on_shapes()
    visualize_predictions(model, x, y, num_examples=3)
