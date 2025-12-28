
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive convolutions with ReLU activation"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


def center_crop(enc_feat, target_feat):
    """Crop encoder features to match decoder feature size"""
    _, _, h, w = target_feat.shape
    _, _, H, W = enc_feat.shape

    dh = (H - h) // 2
    dw = (W - w) // 2

    return enc_feat[:, :, dh:dh + h, dw:dw + w]


class UpConv(nn.Module):
    """Upsampling followed by double convolution"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x_dec, x_enc):
        x_dec = self.up(x_dec)
        x_enc = center_crop(x_enc, x_dec)
        x = torch.cat([x_enc, x_dec], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for biomedical image segmentation
    Paper: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, n_channels=1, n_classes=2):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder (downsampling path)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = DoubleConv(n_channels, 64)
        self.down_conv2 = DoubleConv(64, 128)
        self.down_conv3 = DoubleConv(128, 256)
        self.down_conv4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (upsampling path)
        self.up_conv1 = UpConv(1024, 512)
        self.up_conv2 = UpConv(512, 256)
        self.up_conv3 = UpConv(256, 128)
        self.up_conv4 = UpConv(128, 64)

        # Final 1x1 convolution
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down_conv1(x)           # 64 channels
        x2 = self.down_conv2(self.max_pool(x1))   # 128 channels
        x3 = self.down_conv3(self.max_pool(x2))   # 256 channels
        x4 = self.down_conv4(self.max_pool(x3))   # 512 channels

        # Bottleneck
        x5 = self.bottleneck(self.max_pool(x4))   # 1024 channels

        # Decoder with skip connections
        x = self.up_conv1(x5, x4)         # 512 channels
        x = self.up_conv2(x, x3)          # 256 channels
        x = self.up_conv3(x, x2)          # 128 channels
        x = self.up_conv4(x, x1)          # 64 channels

        # Output
        x = self.out_conv(x)              # n_classes channels

        return x


def init_weights(m):
    """Initialize weights using Kaiming (He) initialization"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# Test the model
if __name__ == "__main__":
    model = UNet(n_channels=1, n_classes=2)
    model.apply(init_weights)

    # Original U-Net input size
    x = torch.randn(1, 1, 572, 572)
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
