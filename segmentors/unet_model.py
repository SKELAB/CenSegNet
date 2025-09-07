import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Dict, Any

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_size),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(UNet, self).__init__()
        
        # Extract basic configuration
        self.n_channels = config['input_channels']
        self.n_classes = config['output_classes']
        self.bilinear = config.get('bilinear', False)
        self.features = config.get('features', [64, 128, 256, 512, 1024])
        
        # Input convolution
        self.inc = DoubleConv(self.n_channels, self.features[0])
        
        # Downsampling path
        self.down_layers = nn.ModuleList()
        for i in range(len(self.features)-1):
            self.down_layers.append(Down(self.features[i], self.features[i+1]))
        
        # Upsampling path
        self.up_layers = nn.ModuleList()
        for i in range(len(self.features)-1, 0, -1):
            self.up_layers.append(
                Up(self.features[i], self.features[i-1], self.bilinear)
            )
        
        # Output convolution
        self.outc = OutConv(self.features[0], self.n_classes)
        
        # Optional checkpoint usage
        self.use_checkpointing = config.get('use_checkpointing', False)

    def forward(self, x):
        # Store intermediate outputs for skip connections
        features = []
        
        # Initial convolution
        x = self.inc(x)
        features.append(x)
        
        # Downsampling path
        for down in self.down_layers:
            x = down(x)
            features.append(x)
        
        # Upsampling path with skip connections
        for up, skip_feature in zip(self.up_layers, features[-2::-1]):
            x = up(x, skip_feature)
        
        # Output convolution
        return self.outc(x)

    def enable_checkpointing(self):
        if self.use_checkpointing:
            self.inc = torch.utils.checkpoint.checkpoint(self.inc)
            for i, layer in enumerate(self.down_layers):
                self.down_layers[i] = torch.utils.checkpoint.checkpoint(layer)
            for i, layer in enumerate(self.up_layers):
                self.up_layers[i] = torch.utils.checkpoint.checkpoint(layer)
            self.outc = torch.utils.checkpoint.checkpoint(self.outc)

def load_model_from_config(config_path: str) -> UNet:
    """Load model configuration from yaml file and create model instance"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return UNet(config)