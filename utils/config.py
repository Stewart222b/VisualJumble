from typing import *
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import *
import models
from utils.io import *
from utils import param

def get_device(device_name: str, kwarg: str=None) -> torch.device:
    """获取 PyTorch 设备。

    根据提供的设备名称返回相应的 PyTorch 设备。

    Args:
        device_name (str): Device name, can be: | 设备名称，可以是：'mps'、'cuda' 或 'cpu'。
        kwarg (str, optional): If provided, this device name will be prioritized. | 如果提供，则优先使用此设备名称。

    Returns:
        torch.device: Returns the corresponding PyTorch device. | 返回相应的 PyTorch 设备。

    Raises:
        Warning: A warning message will be issued if MPS or CUDA is not available. | 如果 MPS 或 CUDA 不可用，将会发出警告信息。

    """
    if kwarg:
        device_name = kwarg

    if device_name == 'mps':
        # Check that MPS is available
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(Warning("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                    " | MPS 不可用，因为当前的 PyTorch 安装未启用 MPS。"))
            else:
                print(Warning("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                    " | MPS 不可用，因为当前的 MacOS 版本不是 12.3+ 或设备不支持 MPS。")) 
            device = torch.device('cpu')
        else:
            device = torch.device('mps') # using cuda

    elif device_name == 'cuda':
        if not torch.cuda.is_available():
            print(Warning("CUDA is not available. Using CPU. | CUDA 不可用，使用 CPU。"))
            device = torch.device('cpu')
        else:
            print(Info("CUDA is available. You can use GPU. | CUDA 可用，可以使用 GPU。"))
            print(Info(f"Number of available GPUs: | 可用的 GPU 数量：{torch.cuda.device_count()}"))
            print(Info(f"Current GPU: | 当前 GPU 名称：{torch.cuda.get_device_name(torch.cuda.current_device())}"))
            device = torch.device('cuda') # using cuda
    elif device_name == 'cpu':
        device = torch.device('cpu') # using cpu
    else:
        print(Warning(f'Invalid device name: {device_name}, use cpu instead | 设备名称无效，使用 CPU。'))
        device = torch.device('cpu')

    print(Info(f'Using device: | 正在使用设备：{device}'))

    return device


def get_model(model_name: str, kwarg: str=None) -> nn.Module:
    if kwarg:
        model_name = kwarg
    
    if model_name == 'lenet':
        model = models.MyLeNet()
    elif model_name == 'alexnet':
        model = models.MyAlexNet()
    elif model_name[:3] == 'vgg':
        model = models.MyVGG(model_name)
    else:
        print(Error(f'please enter a valid model or network: {param.available_net}'))
        exit()

    if not model:
        return -1

    return model


def get_epoch(epoch: int, kwarg: int=None) -> int:
    if kwarg:
        epoch = kwarg

    if epoch < 1:
        print(Error(f'Invalid epoch: {epoch}.'))
        exit()

    return epoch


def get_batch(batch: int, kwarg: int=None) -> int:
    if kwarg:
        batch = kwarg

    if batch < -1 or batch == 0:
        print(Error(f'Invalid batch size: {batch}. i.e. batch -> (0, inf), -1 for auto-batch'))
        exit()
    
    if batch == -1:
        print(Warning('Using auto-batch!'))

    return batch


def get_lr(lr: float, kwarg: float=None) -> float:
    if kwarg:
        lr = kwarg
    
    if lr <= 0 or lr >= 1:
        print(Error(f'Invalid learning rate: {lr}. i.e. lr -> (0, 1)'))
        exit()
    
    return lr


def get_optim(optimizer, model, lr, kwarg) -> Optimizer:
    return SGD(model.parameters(), lr=lr)


def get_criterion(criterion, kwarg) -> nn.Module:
    return nn.CrossEntropyLoss()


def load_data(dataset: str, batch: int, kwarg) -> Tuple[DataLoader, DataLoader]:
    # 数据转换：将图像转换为张量并进行标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 归一化
        transforms.Resize((224, 224))
    ])
    # 加载训练和测试数据集
    if dataset == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        print(Error(''))
        exit()

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    return train_loader, val_loader