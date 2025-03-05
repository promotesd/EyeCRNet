import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import os
import numpy as np
import tqdm
import model
import torch.nn.init as init
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据集定义
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
        label_path = os.path.join(self.label_dir, os.path.splitext(self.images[idx])[0] + ".npy")
        
        image = Image.open(img_path).convert("L")
        label = np.load(label_path)
        
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)

        return image, label


# 数据增强
transform = transforms.Compose([
    transforms.Resize((400, 640)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.RandomResizedCrop(size=(400, 640), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

train_dataset = ds(image_dir="/root/autodl-tmp/dataset/openEDS2019_segmentation/train/images",
                   label_dir='/root/autodl-tmp/dataset/openEDS2019_segmentation/train/labels',
                   transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16)

# Dice 损失
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)        
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss

# 焦点损失
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def initial_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# 模型、损失函数和优化器
model = model.Unet(1, 4).to(device)
initial_weight(model)

def class_weight(train_loader, num_class=4):
    total_pixel = 0
    class_counts = torch.zeros(num_class, dtype=torch.long, device=device)
    
    for images, labels in train_loader:
        labels = labels.view(-1)
        total_pixel += labels.size(0)
        class_counts += torch.bincount(labels, minlength=num_class).to(device=device)

    class_weights = total_pixel / (class_counts * num_class)
    return class_weights.float()

class_weights = class_weight(train_loader).to(device)
dice_loss = DiceLoss().to(device)
focal_loss = FocalLoss().to(device)
ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

# Directories to save models and epoch-loss plots
checkpoint_dir = "/root/autodl-tmp/code/EyeCRNet/train_model_gelu"
epoch_plot_dir = "/root/autodl-tmp/code/EyeCRNet/epoch"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(epoch_plot_dir, exist_ok=True)

# Training loop with model saving and loss tracking
num_epoch = 150
# lowest_loss = float('inf')
# last_saved_model_path = None

epoch_list = []
loss_list = []

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with tqdm.tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{num_epoch}")

        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss_ce = ce_loss(outputs, labels.long())
            loss_dice = dice_loss(outputs, labels)
            loss_focal = focal_loss(outputs, labels)

            loss = loss_ce + loss_dice + loss_focal

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))

            # Collect predictions and labels for IoU and F1 calculations
            preds = outputs.argmax(dim=1).cpu().numpy().flatten()
            labels = labels.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate average loss for the epoch
    avg_epoch_loss = running_loss / len(train_loader)
    epoch_list.append(epoch + 1)
    loss_list.append(avg_epoch_loss)

    # Calculate IoU and F1 scorescree
    iou = jaccard_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.6f}, IoU = {iou:.4f}, F1 = {f1:.4f}")

    
    # if avg_epoch_loss < lowest_loss:
    #     lowest_loss = avg_epoch_loss
    #     if last_saved_model_path:
    #         os.remove(last_saved_model_path)
        
    #     last_saved_model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}_loss_{lowest_loss:.4f}.pth")
    #     torch.save(model.state_dict(), last_saved_model_path)
    #     print(f"Saved model with loss {lowest_loss:.4f} at epoch {epoch+1}")

    # Plot and save the epoch-loss graph for the current epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, loss_list, marker='o', label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Loss")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(epoch_plot_dir, f"epoch_{epoch+1}_loss_curve.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory

    # Step the scheduler
    scheduler.step(avg_epoch_loss)
