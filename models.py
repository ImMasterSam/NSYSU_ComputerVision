import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size=128*128, output_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.layer1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(128, output_classes)

    def forward(self, x):
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.layer1(x)
        x = self.fc2(x)

        return x