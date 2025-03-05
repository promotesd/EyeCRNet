import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import model
import numpy as np
import tqdm
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ds(Dataset):
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

        # 重新映射标签值
        label = torch.tensor(label, dtype=torch.long)
        label = np.where(label < 60, 0, label)
        label = np.where((label >= 60) & (label < 120), 1, label)
        label = np.where((label >= 120) & (label < 180), 2, label)
        label = np.where(label >= 180, 3, label)

        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label, dtype=torch.long)
            # label = label.permute(1, 0)

        return image, label

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.RandomResizedCrop(size=(480, 640), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

train_dataset = ds(image_dir="/root/autodl-tmp/dataset/L5/train/images", label_dir='/root/autodl-tmp/dataset/L5/train/labels', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2)

def initial_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

model = model.Unet(1, 4).to(device)  # 将模型移到 GPU
initial_weight(model)

def class_weight(train_loader, num_class=4):
    total_pixel = 0
    class_counts = np.zeros(num_class)

    for images, labels in train_loader:
        labels = labels.view(-1)
        total_pixel += labels.size(0)
        class_counts += np.bincount(labels.cpu().numpy(), minlength=num_class)
    class_weights = total_pixel / (class_counts * num_class)
    return torch.tensor(class_weights, dtype=torch.float32).to(device)

class_weights = class_weight(train_loader).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# scheduler = StepLR(optimizer=optimizer, step_size=8, gamma=0.3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

num_epoch = 100
lowest_loss = float('inf')  # 初始化最低损失为无穷大

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0

    with tqdm.tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{num_epoch}")

        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            
            loss = criterion(output, labels.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=running_loss/(tepoch.n+1))
            print(tepoch)

    scheduler.step()

    average_loss = running_loss / len(train_loader)
    if average_loss < lowest_loss:
        lowest_loss = average_loss  # 更新最低损失
        torch.save(model.state_dict(), '/root/autodl-tmp/code/EyeCRNet/train_model' + f'model_epoch_{epoch+1}.pth')
        print(f"Model saved after epoch {epoch+1} with improved loss {lowest_loss}")
    else:
        print(f"No improvement in epoch {epoch+1}, not saving model.")

