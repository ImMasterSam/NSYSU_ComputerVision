import torch
import torchvision
import torch.nn.functional as F
import os
from typing import Tuple

class imagenette_dataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        # Get class names and indices
        classes = os.listdir(folder_path)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        # Load all image paths and labels
        self.data: Tuple[str, int] = [] # List of (image_path, label_idx)
        for cls_name in classes:
            cls_folder = os.path.join(folder_path, cls_name)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                self.data.append((img_path, self.class_to_idx[cls_name]))

        # Store transform
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = torchvision.io.read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def train_model(model, train_dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for inputs, targets in train_dataloader:
        # 1. Get inputs and targets from dataloader and move them to device
        inputs, targets = inputs.to(device), targets.to(device)
        # 2. Zero the parameter gradients
        optimizer.zero_grad()
        # 3. Forward pass
        outputs = model(inputs)
        # 4. Compute loss
        loss = F.cross_entropy(outputs, targets.squeeze())
        accuracy = (outputs.argmax(dim=1) == targets).float().sum().item()

        # 5. Backward pass by calling autograd()
        loss.backward()

        # 6. Update parameters using optimizer
        optimizer.step()
        ################################################################################
        # END OF YOUR CODE
        ################################################################################

        running_loss += loss.item() * inputs.size(0)
        running_accuracy += accuracy

    epoch_loss = running_loss / len(train_dataloader.dataset)
    epoch_accuracy = running_accuracy / len(train_dataloader.dataset)
    return epoch_loss, epoch_accuracy


def evaluate_model(model, val_dataloader, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            # 1. Get inputs and targets from dataloader and move them to device
            inputs, targets = inputs.to(device), targets.to(device)

            # 2. Forward pass
            outputs = model(inputs)

            # 3. Compute loss
            loss = F.cross_entropy(outputs, targets)
            accuracy = (outputs.argmax(dim=1) == targets).float().sum().item()

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += accuracy

    epoch_loss = running_loss / len(val_dataloader.dataset)
    epoch_accuracy = running_accuracy / len(val_dataloader.dataset)
    return epoch_loss, epoch_accuracy