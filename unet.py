import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder - only 2 layers
        self.enc1 = self._block(3, 32)    # Changed from 32 to 3 for RGB input
        self.enc2 = self._block(32, 64)
        
        # Bottleneck
        self.bottleneck = self._block(64, 128)
        
        # Decoder - only 2 layers
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128,64)
        self.up2 = nn.ConvTranspose2d(64,32,kernel_size=2, stride=2)
        self.dec2 = self._block(64,32)
        
        # Output
        self.out = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)
    
    def _block(self, in_channels, out_channels, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder - only 2 layers
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc2))
        
        # Decoder - only 2 layers
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat((dec1, enc2), dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)
        
        # Output
        out = self.out(dec2)
        return self.sigmoid(out)
#########################################################################################################

class Unet4(nn.Module):
    def __init__(self):
        super(Unet4, self).__init__()
        # Encoder
        self.enc1 = self._block(3, 16)
        self.enc2 = self._block(16, 32)
        self.enc3 = self._block(32, 64)
        self.enc4 = self._block(64, 128)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(128, 256)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self._block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self._block(64, 32)
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec4 = self._block(32, 16)

        # Output
        self.out = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec1 = self.dec1(dec1)

        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.dec2(dec2)

        dec3 = self.up3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)

        dec4 = self.up4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec4 = self.dec4(dec4)

        # Output
        out = self.out(dec4)
        return self.sigmoid(out)
