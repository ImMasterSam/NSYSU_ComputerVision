import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size=128*128, output_classes=2):
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
    
class LeNet_5(nn.Module):
    def __init__(self, input_size=128*128, output_classes=2, in_channel=1):
        super().__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channel, 6, 5, padding=2), 
            nn.Sigmoid(),       
            nn.MaxPool2d(2, stride=2),  
            nn.Conv2d(6, 16, 5, stride=1, padding=2),       
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(16*(input_size // 16), 120),         
            nn.Sigmoid(),
            nn.Linear(120, 84),          
            nn.Sigmoid(),
            nn.Linear(84, output_classes)            
        )

    def forward(self, x):
        
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)

        return x