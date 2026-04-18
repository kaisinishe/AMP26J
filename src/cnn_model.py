import torch
import torch.nn as nn
import torch.nn.functional as F

class OnsetCNN(nn.Module):
    def __init__(self, window_size=31): # Increased default window
        super(OnsetCNN, self).__init__()
        
        # Layer 1: Captures fine spectral details
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2: Captures broader patterns
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3: Deep features
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3) # Prevent overfitting
        
        # After 2 pooling layers: 
        # 80x31 -> 40x15 -> 20x7
        self.fc1 = nn.Linear(128 * 20 * 7, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x))) # No pooling on the last conv to keep resolution
        
        x = x.view(x.size(0), -1) 
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x