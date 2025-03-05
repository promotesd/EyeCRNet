import torch.onnx
from model import Unet
model = Unet(1, 4)  # 例如，输入通道为1，输出通道为1
dummy_input = torch.randn(16, 1, 400, 640)  # 示例输入
torch.onnx.export(model, dummy_input, "unet_model.onnx")
