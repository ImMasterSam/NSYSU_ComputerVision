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
    def __init__(self, input_size=224, output_classes=4, in_channels=3):
        super().__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2), 
            nn.Sigmoid(), 
            nn.BatchNorm2d(6),      
            nn.MaxPool2d(2, stride=2),  
            nn.Conv2d(6, 16, 5, stride=1, padding=2),       
            nn.Sigmoid(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.cnn_model(dummy)
            flatten_dim = out.view(1, -1).size(1)

        self.fc_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 120),         
            nn.Sigmoid(),
            nn.Linear(120, 84),          
            nn.Sigmoid(),
            nn.Linear(84, output_classes)            
        )

    def forward(self, x):
        
        x = self.cnn_model(x)
        x = self.fc_model(x)

        return x
    
class AlexNet(nn.Module):
    def __init__(self, input_size=224, output_classes=4, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # 計算 flatten 維度
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.features(dummy)
            flatten_dim = out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGGNet(nn.Module):
    cfgs = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, vgg_name='VGG16', output_classes=4, in_channels=3, input_size=224):
        super().__init__()
        self.cnn_model = self.make_layers(self.cfgs[vgg_name], in_channels)
        # 計算最後一層 feature map 的大小
        dummy = torch.zeros(1, in_channels, input_size, input_size)
        with torch.no_grad():
            out = self.cnn_model(dummy)
        flatten_dim = out.view(1, -1).size(1)
        self.fc_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_classes)
        )

    def forward(self, x):

        x = self.cnn_model(x)
        x = self.fc_model(x)
        
        return x

    def make_layers(self, cfg, in_channels):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                in_channels = v
        return nn.Sequential(*layers)
    
class CustomCNN(nn.Module):
    def __init__(self, output_classes=4, in_channels=3, input_size=224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=8, stride=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.features(dummy)
            flatten_dim = out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, output_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x