"""
Modern U-Net Architecture with Padding
Προσαρμοσμένο για να διατηρεί σταθερές διαστάσεις (256x256)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        layers = [
            # Το padding=1 => δεν μικραίνει η εικόνα!
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
            layers.insert(4, nn.BatchNorm2d(out_channels))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=4, batch_norm=True):
        super(UNet, self).__init__()
        self.name = 'UNet'
        self.n_classes = n_classes
        
        # Encoder (Κατεβαίνουμε)
        self.inc = DoubleConv(1, 64, batch_norm)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, batch_norm))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, batch_norm))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512, batch_norm))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024, batch_norm))
        
        # Decoder (Ανεβαίνουμε)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512, batch_norm)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256, batch_norm)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128, batch_norm)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64, batch_norm)
        
        # Τελικό Layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Contract
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Expand 1
        x = self.up1(x5)
        # Safe Concat: Εξασφαλίζει ότι το x ταιριάζει ΑΠΟΛΥΤΑ με το x4 
        x = F.interpolate(x, size=(x4.shape[2], x4.shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        # Expand 2
        x = self.up2(x)
        x = F.interpolate(x, size=(x3.shape[2], x3.shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        # Expand 3
        x = self.up3(x)
        x = F.interpolate(x, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        # Expand 4
        x = self.up4(x)
        x = F.interpolate(x, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=False)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits