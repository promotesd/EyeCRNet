import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import model
import numpy as np

# 加载训练好的模型权重
model_dict_path = '/root/autodl-tmp/code/EyeCRNet/gelu1tun/model_epoch_87_loss_0.1569.pth'

# 初始化模型
model = model.Unet(1, 4)
model.load_state_dict(torch.load(model_dict_path))
model.eval()

# 加载并预处理图像
image_path = "/root/autodl-tmp/dataset/openEDS2019_segmentation/train/images/000000.png"
image = Image.open(image_path).convert("L")  # 转换为灰度模式

# 将图像转换为 NumPy 数组并规范化为 0-1 之间的值
image_np = np.array(image, dtype=np.float32) / 255.0  # 归一化

# 转换为 torch.FloatTensor
input_image = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 增加通道维度和 batch 维度

# 推理
with torch.no_grad():
    output = model(input_image)
    predicted_classes = torch.argmax(output, dim=1)  # 获取每个像素的类别
    print(torch.unique(predicted_classes))  # 打印输出类别的唯一值

# 可视化输出
output_image = predicted_classes.squeeze(0).cpu().numpy()  # 移除 batch 维度并转换为 numpy
print(output_image.shape)
plt.imshow(output_image, cmap='gray')
plt.title("Segmentation Result")
plt.show()

# 保存输出图像
output_path = "/root/autodl-tmp/code/EyeCRNet/test.jpg"
plt.imsave(output_path, output_image, cmap="gray")


