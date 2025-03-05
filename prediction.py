import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import model
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
import os

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
model_dict_path = "/root/autodl-tmp/code/EyeCRNet/gelu1tun/model_epoch_87_loss_0.1569.pth"
unet_model = model.Unet(1, 4).to(device)
unet_model.load_state_dict(torch.load(model_dict_path))
unet_model.eval()

# 定义数据集类
class DS(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, limit=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        if limit:
            self.images = self.images[:limit]  # 限制图像数量

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

# 定义图像变换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 创建数据集和数据加载器
validation_dataset = DS(
    image_dir="/root/autodl-tmp/dataset/openEDS2019_segmentation/validation/images", 
    label_dir="/root/autodl-tmp/dataset/openEDS2019_segmentation/validation/labels", 
    transform=transform, 
    limit=None
)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# 定义评估指标计算函数
def compute_metrics(preds, labels, num_classes):
    preds_tensor = torch.tensor(preds)
    labels_tensor = torch.tensor(labels)

    pixel_accuracy = (preds_tensor == labels_tensor).float().mean().item()
    mean_accuracy = 0
    mean_iou = 0
    mean_f1 = 0
    
    for cls in range(num_classes):
        tp = ((preds_tensor == cls) & (labels_tensor == cls)).sum().item()
        fp = ((preds_tensor == cls) & (labels_tensor != cls)).sum().item()
        fn = ((preds_tensor != cls) & (labels_tensor == cls)).sum().item()
        if tp + fp > 0:
            mean_accuracy += tp / (tp + fp)
        if tp + fn > 0:
            mean_f1 += tp / (tp + fn)
        if tp + fp + fn > 0:
            mean_iou += tp / (tp + fp + fn)

    mean_accuracy /= num_classes
    mean_f1 /= num_classes
    mean_iou /= num_classes

    return pixel_accuracy, mean_accuracy, mean_f1, mean_iou

# 执行预测并计算指标
all_preds, all_labels, all_images = [], [], []
for images, labels in tqdm.tqdm(validation_loader):
    images = images.to(device)
    with torch.no_grad():
        outputs = unet_model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    all_images.extend(images.cpu().numpy())  # Ensure images are on CPU
    all_preds.extend(preds)
    all_labels.extend(labels.numpy())

# 计算指标
metrics = compute_metrics(all_preds, all_labels, num_classes=4)

# 打印并保存评估指标
metrics_output_path = "/root/autodl-tmp/code/EyeCRNet/newresult/EDS2019validation_n/metrics.txt"
with open(metrics_output_path, 'w') as f:
    f.write(f'Pixel Accuracy: {metrics[0]:.4f}\n')
    f.write(f'Mean Accuracy: {metrics[1]:.4f}\n')
    f.write(f'Mean F1 Score: {metrics[2]:.4f}\n')
    f.write(f'Mean IoU: {metrics[3]:.4f}\n')

print(f"总图片数: {len(all_images)}")
print(f"总预测数: {len(all_preds)}")
print(f"总标签数: {len(all_labels)}")

# 可视化预测结果并保存图片
def visualize_predictions(images, preds, labels, save_dir, num_samples=5000):
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title('Predicted Mask')
        plt.imshow(preds[i], cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title('True Mask')
        plt.imshow(labels[i], cmap='gray')
        plt.savefig(os.path.join(save_dir, f'result_{i}.png'))
        plt.close()

visualize_predictions(all_images, all_preds, all_labels, "/root/autodl-tmp/code/EyeCRNet/newresult/EDS2019validation_n")
