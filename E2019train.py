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
        # image=np.array(image)
        # image=Image.fromarray(image)
        label = np.load(label_path)#class 'numpy.ndarray
        
        label = torch.tensor(label, dtype=torch.long) # <class 'PIL.Image.Image'>, label type: <class 'torch.Tensor'>
        

        if self.transform:
            # print(f"Before transform, image type: {type(image)}")# <class 'PIL.Image.Image'>
            image = self.transform(image)
            
        

        return image, label


# 数据增强
transform = transforms.Compose([
    transforms.Resize((400, 640)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=(-30, 30)),
    transforms.RandomResizedCrop(size=(400, 640), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

train_dataset = ds(image_dir="/root/autodl-tmp/dataset/openEDS2019_segmentation/train/images", label_dir='/root/autodl-tmp/dataset/openEDS2019_segmentation/train/labels', transform=transform)
# train_dataset = ds(image_dir="/root/autodl-tmp/dataset/openEDS2019_segmentation/train/images", label_dir='/root/autodl-tmp/dataset/openEDS2019_segmentation/train/labels')

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
            # 使用He初始化，适用于GELU激活
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 虽然是GELU，这里使用'relu'也是可接受的
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # 批量归一化层的权重初始化为1，偏置初始化为0
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # 对全连接层使用Xavier初始化可能更合适
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# 模型、损失函数和优化器
model = model.Unet(1, 4).to(device)
initial_weight(model)

def class_weight(train_loader, num_class=4):
    total_pixel = 0
    class_counts = torch.zeros(num_class, dtype=torch.long, device=device)  # Initialize on the correct device
    
    for images, labels in train_loader:
       
        
        labels = labels.view(-1)  # Flatten the labels
        total_pixel += labels.size(0)
        class_counts += torch.bincount(labels, minlength=num_class).to(device=device)  # Directly use torch.bincount

    class_weights = total_pixel / (class_counts * num_class)
    return class_weights.float()  # Ensure it's floating point for later computations

class_weights = class_weight(train_loader).to(device)
dice_loss = DiceLoss().to(device)
focal_loss = FocalLoss().to(device)
ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

# 训练循环
num_epoch = 100
lowest_loss = float('inf')
previous_loss = float('inf')
patience_counter = 0
last_saved_model_path = None

# 初始化最低损失
lowest_loss = float('inf')
last_saved_model_path = None

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0

    with tqdm.tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{num_epoch}")

        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss_ce = ce_loss(outputs, labels.long())
            loss_dice = dice_loss(outputs, labels)
            loss_focal = focal_loss(outputs, labels)

            # 损失求和
            loss = loss_ce + loss_dice + loss_focal

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))

    # 计算当前 epoch 的平均损失
    avg_epoch_loss = running_loss / len(train_loader)

    # 检查当前损失是否小于历史最低损失
    if avg_epoch_loss < lowest_loss:
        # 更新最低损失
        lowest_loss = avg_epoch_loss
        
        # 如果有上一回合的模型文件，删除它
        if last_saved_model_path:
            os.remove(last_saved_model_path)
        
        # 保存当前回合的模型参数
        last_saved_model_path = f"/root/autodl-tmp/code/EyeCRNet/train_model_gelu/model_epoch_{epoch+1}_loss_{avg_epoch_loss:.4f}.pth"
        torch.save(model.state_dict(), last_saved_model_path)
        print(f"Saved model with loss {avg_epoch_loss:.4f} at epoch {epoch+1}")
    else:
        print(f"Loss did not improve from {lowest_loss:.4f} at epoch {epoch+1}")

    # 调整学习率
    scheduler.step(avg_epoch_loss)
    