import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)   # 64x64 → 64x64
        self.pool1 = nn.MaxPool2d(2, 2)                            # 64x64 → 32x32
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32 → 32x32
        self.pool2 = nn.MaxPool2d(2, 2)                            # 32x32 → 16x16
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # 16x16 → 16x16
        self.pool3 = nn.MaxPool2d(2, 2)                            # 16x16 → 8x8
        
        self.flatten = nn.Flatten()                                # 256 * 8 * 8 = 16384
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 200)                             # 200 classi

    def forward(self, x):
        x = self.pool1(self.conv1(x).relu())
        x = self.pool2(self.conv2(x).relu())
        x = self.pool3(self.conv3(x).relu())
        x = self.flatten(x)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x