import torch
import numpy as np
import model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 加载训练好的模型权重
model_dict_path = '/root/autodl-tmp/code/EyeCRNet/gelu1tun/model_epoch_87_loss_0.1569.pth'

# 初始化模型
model = model.Unet(1, 4)
model.load_state_dict(torch.load(model_dict_path))

# 计算参数数量
total_params = sum(p.numel() for p in model.parameters())

# 计算模型大小（以MB为单位）
total_size = sum(p.element_size() * p.numel() for p in model.parameters())
total_size_MB = total_size / (1024 ** 2)

print(f'Total Parameters: {total_params}')   #31.61M
print(f'Total Size (MB): {total_size_MB:.2f}')  #120.59MB
