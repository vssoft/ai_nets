# unet_model.py
#
# This file defines the U-Net neural network model used for IMAGE SEGMENTATION.
# Image segmentation means: for every pixel in the input image, the model predicts
# whether that pixel belongs to the object of interest (here: tumor) or background.
#
# U-Net is a popular architecture for medical image segmentation because it works
# well even with relatively small datasets.

# -------------------------------------------------------------------------
# U-NET (Simple Diagram with Sizes)
#
# Each block shows:  [Channels, Height x Width]
#
# Input
#   [  1, 128x128]
#        |
#        v
# Encoder (Downsampling path)              Decoder (Upsampling path)
#   [ 64, 128x128]  ----skip---->      [ 64, 128x128]
#        |                               ^
#      pool                             up
#        v                               |
#   [128,  64x64 ]  ----skip---->      [128,  64x64 ]
#        |                               ^
#      pool                             up
#        v                               |
#   [256,  32x32 ]  ----skip---->      [256,  32x32 ]
#        |                               ^
#      pool                             up
#        v                               |
#   [512,  16x16 ]  ----skip---->      [512,  16x16 ]
#        |                               ^
#      pool                             up
#        v                               |
# Bottleneck
#   [1024,  8x8 ]
#
# Output
#   [  1, 128x128]  (logits; apply sigmoid -> probabilities)
#
# Notes:
# - "pool" = MaxPool2d(2): reduces size by 2 (128->64->32->...)
# - "up"   = ConvTranspose2d: increases size by 2
# - "skip" = concatenation of encoder features into decoder (keeps detail)
# -------------------------------------------------------------------------

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    A common building block in U-Net:
    Two consecutive "Conv2D + ReLU" layers.

    Why do we use convolution layers?
    - They detect patterns in images such as edges, textures, shapes, etc.
    - Each conv layer learns "filters" (small pattern detectors) automatically.

    Why two convolutions in a row?
    - This allows the network to learn richer features at each resolution level.
    """

    def __init__(self, in_ch, out_ch):
        """
        Parameters:
        - in_ch:  number of input channels (e.g. 1 for grayscale image)
        - out_ch: number of output channels (feature maps produced by this block)

        In segmentation networks, the number of channels is like the "depth" of feature
        representation: more channels => more learned features.
        """
        super().__init__()

        # nn.Sequential means: apply layers one after another in order.
        self.block = nn.Sequential(
            # First convolution:
            # kernel_size=3 means a 3x3 filter is used.
            # padding=1 keeps the image size the same (so output H and W stay unchanged).
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),

            # ReLU activation:
            # Adds non-linearity so the model can learn complex patterns.
            nn.ReLU(inplace=True),

            # Second convolution (same idea as above):
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        forward() defines how data flows through this block.
        PyTorch calls this automatically when you do:
            output = block(input)
        """
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net model for binary segmentation.

    U-Net structure:
    - Encoder (downsampling path):
        Compresses the image into smaller and smaller representations.
        Learns high-level features (what and where).
    - Bottleneck:
        The deepest part, contains the most abstract features.
    - Decoder (upsampling path):
        Expands back to full resolution.
        Produces a mask the same size as the input image.
    - Skip connections:
        Copy feature maps from encoder to decoder so we don’t lose spatial detail.

    Input:
      x shape = (B, 1, H, W)
      B = batch size
      1 = grayscale channel
      H, W = height and width (e.g., 128 x 128)

    Output:
      out shape = (B, 1, H, W)
      These are "logits" (raw values, not probabilities).
      Later in training:
          sigmoid(logits) -> probabilities between 0 and 1
    """

    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        """
        Parameters:
        - in_channels: number of image channels (1 for grayscale, 3 for RGB)
        - out_channels: number of output channels (1 for binary segmentation)
        - base_filters: number of filters in the first layer (controls model size)

        U-Net grows feature depth as:
            base_filters, 2*base_filters, 4*base_filters, ...
        Typical base_filters values: 32, 64.
        """
        super().__init__()

        # -------------------------
        # Encoder (downsampling path)
        # -------------------------
        # Each encoder step:
        # 1) runs DoubleConv (extract features)
        # 2) runs MaxPool2d (reduce spatial size by 2)
        #
        # Example:
        # 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        self.conv1 = DoubleConv(in_channels, base_filters)              # output channels: 64
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(base_filters, base_filters * 2)         # output channels: 128
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(base_filters * 2, base_filters * 4)     # output channels: 256
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(base_filters * 4, base_filters * 8)     # output channels: 512
        self.pool4 = nn.MaxPool2d(2)

        # -------------------------
        # Bottleneck
        # -------------------------
        # Deepest part of the network.
        # The spatial size is smallest here (e.g., 8x8),
        # but the number of feature channels is largest (e.g., 1024).
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)  # output channels: 1024

        # -------------------------
        # Decoder (upsampling path)
        # -------------------------
        # Each decoder step:
        # 1) Upsample using ConvTranspose2d (increases spatial size x2)
        # 2) Concatenate with matching encoder output (skip connection)
        # 3) Apply DoubleConv to refine the combined features

        # Decoder stage 1 (1024 -> 512)
        self.upconv1 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(base_filters * 16, base_filters * 8)  # concat doubles channels

        # Decoder stage 2 (512 -> 256)
        self.upconv2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(base_filters * 8, base_filters * 4)

        # Decoder stage 3 (256 -> 128)
        self.upconv3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(base_filters * 4, base_filters * 2)

        # Decoder stage 4 (128 -> 64)
        self.upconv4 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(base_filters * 2, base_filters)

        # -------------------------
        # Output layer
        # -------------------------
        # 1x1 convolution converts feature maps into the final output mask channels.
        # Here out_channels=1 for binary segmentation.
        #
        # Output is logits (raw values).
        # Later we apply sigmoid(logits) to get probabilities.
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass: defines how input image flows through the network.
        """

        # -------------------------
        # Encoder forward
        # -------------------------
        # c1..c4 store feature maps used later in skip connections.
        c1 = self.conv1(x)     # features at full resolution (128x128)
        p1 = self.pool1(c1)    # downsample -> 64x64

        c2 = self.conv2(p1)    # 64x64
        p2 = self.pool2(c2)    # downsample -> 32x32

        c3 = self.conv3(p2)    # 32x32
        p3 = self.pool3(c3)    # downsample -> 16x16

        c4 = self.conv4(p3)    # 16x16
        p4 = self.pool4(c4)    # downsample -> 8x8

        # -------------------------
        # Bottleneck
        # -------------------------
        bn = self.bottleneck(p4)  # 8x8 with deep features (1024 channels)

        # -------------------------
        # Decoder forward
        # -------------------------
        # Upsample and concatenate with corresponding encoder features.
        # torch.cat(..., dim=1) concatenates along channel dimension.
        u1 = self.upconv1(bn)               # upsample 8x8 -> 16x16
        cat1 = torch.cat([u1, c4], dim=1)   # skip connection (adds detail)
        c5 = self.conv5(cat1)

        u2 = self.upconv2(c5)               # 16x16 -> 32x32
        cat2 = torch.cat([u2, c3], dim=1)
        c6 = self.conv6(cat2)

        u3 = self.upconv3(c6)               # 32x32 -> 64x64
        cat3 = torch.cat([u3, c2], dim=1)
        c7 = self.conv7(cat3)

        u4 = self.upconv4(c7)               # 64x64 -> 128x128
        cat4 = torch.cat([u4, c1], dim=1)
        c8 = self.conv8(cat4)

        # -------------------------
        # Output logits
        # -------------------------
        # out is the predicted segmentation mask (logits).
        # Apply sigmoid(out) elsewhere to get probabilities.
        out = self.out_conv(c8)

        return out


if __name__ == "__main__":
    # Self-test: verify shapes
    # This is a simple sanity check: input and output should be the same spatial size.
    model = UNet(in_channels=1, out_channels=1, base_filters=64)

    # Create a dummy batch:
    # batch size=2, grayscale channel=1, image size=128x128
    x = torch.randn(2, 1, 128, 128)

    # Forward pass through the model
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)