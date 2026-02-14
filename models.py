import torch
import torch.nn as nn
import torch.nn.functional as F


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
                nn.ReLU(inplace=True),
                nn.Dropout(p =0.2)
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

class LeNet(nn.Module):
    def __init__(self, channels_in=3, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))
        x = self.maxpool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        # Add adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Now we know the size will be 256 * 6 * 6 = 9216
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)  # Apply adaptive pooling
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
