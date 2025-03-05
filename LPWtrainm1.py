import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import os
import model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DS(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("L")
        label_name = os.path.splitext(self.images[idx])[0] + ".npy"
        label_path = os.path.join(self.label_dir, label_name)
        label = np.load(label_path)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.RandomResizedCrop(size=(480, 640), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

train_dataset = DS(image_dir="/path/to/train/images", label_dir='/path/to/train/labels', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(inputs * targets, dim=(2, 3))
        union = torch.sum(inputs + targets, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

def combined_loss(output, labels):
    dice_loss = DiceLoss().to(device)
    focal_loss = FocalLoss().to(device)
    return dice_loss(output, labels) + focal_loss(output, labels)

unet_model = model.Unet(1, 4).to(device)
unet_model.load_state_dict(torch.load("/root/autodl-tmp/code/EyeCRNet/gelu1tun/model_epoch_87_loss_0.1569.pth"))

optimizer = Adam(unet_model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

num_epoch = 100
lowest_loss = float('inf')

for epoch in range(num_epoch):
    unet_model.train()
    running_loss = 0.0
    for images, labels in tqdm.tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        output = unet_model(images)
        loss = combined_loss(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step(running_loss / len(train_loader))
    average_loss = running_loss / len(train_loader)
    if average_loss < lowest_loss:
        lowest_loss = average_loss
        torch.save(unet_model.state_dict(), f'/path/to/save/model_epoch_{epoch+1}.pth')
        print(f"Model saved after epoch {epoch+1} with improved loss {lowest_loss}")

