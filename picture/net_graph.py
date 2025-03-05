import torch
from torchsummary import summary
from model import Unet
# 假设您的 U-Net 模型定义在 `Unet` 类中
model = Unet(in_channels=1, out_channels=4)  # 初始化模型
model.to('cuda')  # 将模型移动到 GPU 上（如果有 GPU）

# 打印模型结构
summary(model, input_size=(1, 400, 640))  # 假设输入是单通道的 256x256 图像