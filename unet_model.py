import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d + ReLU layers, used throughout U-Net.
    """
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
    """
    U-Net model for binary segmentation.

    Input:  (B, 1, H, W)
    Output: (B, 1, H, W)  <-- logits (not sigmoid)
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super().__init__()

        # Encoder
        self.conv1 = DoubleConv(in_channels, base_filters)              # 64
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(base_filters, base_filters * 2)         # 128
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(base_filters * 2, base_filters * 4)     # 256
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(base_filters * 4, base_filters * 8)     # 512
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)  # 1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(base_filters * 16, base_filters * 8)

        self.upconv2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(base_filters * 8, base_filters * 4)

        self.upconv3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(base_filters * 4, base_filters * 2)

        self.upconv4 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(base_filters * 2, base_filters)

        # Output layer (logits)
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

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

        # Output logits
        out = self.out_conv(c8)
        return out


if __name__ == "__main__":
    # Self-test: verify shapes
    model = UNet(in_channels=1, out_channels=1, base_filters=64)
    x = torch.randn(2, 1, 128, 128)  # batch=2, grayscale, 128x128
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
