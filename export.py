from models.train_engine import TrainEngine
from utils.io import *

import torch
import torch.onnx

def convert_to_onnx(pt_model_path, onnx_model_path, input_size=(1, 3, 224, 224)):
    """
    将 PyTorch 模型转换为 ONNX 格式
    | Convert a PyTorch model to ONNX format.

    Args:
        pt_model_path (str): PyTorch 模型的路径 | Path to the PyTorch model (.pt file).
        onnx_model_path (str): 转换后保存的 ONNX 模型路径 | Path to save the converted ONNX model.
        input_size (tuple): 模型输入的尺寸，默认是 (1, 3, 224, 224) | The input size for the model, default is (1, 3, 224, 224).
    """

    # 加载 PyTorch 模型 | Load the PyTorch model
    engine = TrainEngine(model='lprnet', device='cpu')
    model = engine.model
    device = engine.device

    param = torch.load(pt_model_path, map_location=device, weights_only=False)
    model.load_state_dict(param)
    model.eval()  # 切换到评估模式 | Switch to evaluation mode

    # 创建模型的输入张量 | Create a dummy input tensor
    dummy_input = torch.randn(*input_size).to(device)

    # 导出 ONNX 模型 | Export to ONNX model
    torch.onnx.export(
        model,                     # 被导出的模型 | The model to export
        dummy_input,               # 示例输入张量 | Example input tensor
        onnx_model_path,           # 保存的 ONNX 模型路径 | Path to save ONNX model
        export_params=True,        # 是否导出模型的权重参数 | Whether to export model's parameters
        opset_version=11,          # ONNX opset 版本 | ONNX opset version
        do_constant_folding=True,  # 是否执行常量折叠优化 | Whether to perform constant folding for optimization
        input_names=['input'],     # 输入名称 | Input names
        output_names=['output'],   # 输出名称 | Output names
        #dynamic_axes={'input': {0: 'batch_size'},  # 支持动态批量大小 | Support dynamic batch size
        #              'output': {0: 'batch_size'}}
    )

    print(Info(f"模型已成功导出到 {onnx_model_path} | Model has been successfully exported to {onnx_model_path}"))

if __name__ == '__main__':
    # 示例调用 | Example usage
    pt_model_path = "/Users/Larry/Projects/LPRNet_Pytorch-master/weights/Final_LPRNet_model.pth"
    onnx_model_path = "/Users/Larry/Projects/LPRNet_Pytorch-master/weights/model.onnx"
    convert_to_onnx(pt_model_path, onnx_model_path, input_size=(1, 3, 24, 94))