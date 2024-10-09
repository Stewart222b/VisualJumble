from typing import *
import cv2
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import *
from PIL import Image
import models
from utils.io import *
from utils.param import DEVICES, MODELS, DATASETS, OPTIMIZERS, CRITERIONS

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

    if device_name not in DEVICES:
        print(Warning(f'Invalid device name: {device_name}, use cpu instead | 设备名称无效，使用 CPU。'))
        device = torch.device('cpu')

    elif device_name == 'mps':
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

    print(Info(f'Using device: | 正在使用设备：{device}'))

    return device


def get_model(model_name: str, return_info: bool=False, kwarg: str=None):
    """
    Get a model and optionally return its information.

    Args:
        model_name (str): The name of the model to retrieve. 模型名称
        return_info (bool, optional): Whether to return model information. Defaults to False. 是否返回模型信息，默认为 False。
        kwarg (str, optional): Override for model_name if provided. Defaults to None. 如果提供，将覆盖 model_name，默认为 None。

    Returns:
        model: The initialized model object. 初始化的模型对象
        info (optional): The model information if return_info is True. 如果 return_info 为 True，返回模型信息
    """
    
    if kwarg:
        model_name = kwarg

    if model_name not in MODELS:
        print(Error(f'please enter a valid model or network: {MODELS}'))
        exit()
    
    if model_name == 'lenet':
        model = models.MyLeNet()
        info = models.LeNetInfo()

    elif model_name == 'alexnet':
        model = models.MyAlexNet()
        info = models.AlexNetInfo()

    elif model_name[:3] == 'vgg':
        model = models.MyVGG(model_name)
        info = models.VGGInfo()

    elif model_name == 'lprnet':
        model = models.MyLPRNet()
        info = models.LPRNetInfo()

    if not model:
        return -1

    if return_info and info:
        return model, info
    else:
        return model


def get_epoch(epoch: int, kwarg: int=None) -> int:
    """
    Get the number of epochs for training.

    Args:
        epoch (int): The number of epochs. 训练的轮数
        kwarg (int, optional): Override for epoch if provided. Defaults to None. 如果提供，将覆盖 epoch，默认为 None。

    Returns:
        int: Validated number of epochs. 验证后的轮数
    """

    if kwarg:
        epoch = kwarg

    if epoch < 1:
        print(Error(f'Invalid epoch: {epoch}.'))
        exit()

    return epoch


def get_batch(batch: int, kwarg: int=None) -> int:
    """
    Get the batch size for training.

    Args:
        batch (int): The batch size. 批量大小
        kwarg (int, optional): Override for batch if provided. Defaults to None. 如果提供，将覆盖 batch，默认为 None。

    Returns:
        int: Validated batch size. 验证后的批量大小
    """

    if kwarg:
        batch = kwarg

    if batch < -1 or batch == 0:
        print(Error(f'Invalid batch size: {batch}. i.e. batch -> (0, inf), -1 for auto-batch'))
        exit()
    
    if batch == -1:
        print(Warning('Using auto-batch!'))

    return batch


def get_lr(lr: float, kwarg: float=None) -> float:
    """
    Get the learning rate for training.

    Args:
        lr (float): The learning rate. 学习率
        kwarg (float, optional): Override for lr if provided. Defaults to None. 如果提供，将覆盖 lr，默认为 None。

    Returns:
        float: Validated learning rate. 验证后的学习率
    """

    if kwarg:
        lr = kwarg
    
    if lr <= 0 or lr >= 1:
        print(Error(f'Invalid learning rate: {lr}. i.e. lr -> (0, 1)'))
        exit()
    
    return lr


def get_optim(optimizer, model, lr, kwarg) -> Optimizer:
    """
    Get the optimizer for training.

    Args:
        optimizer: The optimizer type. | 优化器类型
        model: The model to optimize. | 待优化的模型
        lr: The learning rate. | 学习率
        kwarg: Override for optimizer if provided. | 如果提供，将覆盖 optimizer。

    Returns:
        Optimizer: The optimizer object. | 优化器对象
    """

    if kwarg:
        optimizer = kwarg

    if optimizer not in OPTIMIZERS:
        print(Error(f'please enter a valid optimizer: {OPTIMIZERS}'))
        exit()

    return SGD(model.parameters(), lr=lr)


def get_criterion(criterion, kwarg) -> nn.Module:
    """
    Get the loss function for training.

    Args:
        criterion: The loss function type. 损失函数类型
        kwarg: Override for criterion if provided. 如果提供，将覆盖 criterion。

    Returns:
        nn.Module: The loss function object. 损失函数对象
    """

    if kwarg:
        criterion = kwarg

    if criterion not in CRITERIONS:
        print(Error(f'please enter a valid criterion: {CRITERIONS}'))
        exit()
    
    return nn.CrossEntropyLoss()


def load_dataset(dataset: str, batch: int, kwarg) -> Tuple[DataLoader, DataLoader]:
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


def read_image(image_path, 
               transform,
               size=(112, 112), 
               mean=[0.485, 0.456, 0.406], 
               std=[0.229, 0.224, 0.225], 
               mode='RGB'):
    """
    读取单张图片，进行预处理
    
    Args:
        model (torch.nn.Module): 训练好的 PyTorch 模型.
        image_path (str): 图片的路径.
        device (str): 设备类型 ('cpu' 或 'cuda').
    
    Returns:
        torch.Tensor: 模型的输出.
    """
    # 读取和预处理图片
    #image = Image.open(image_path).convert(mode)  # 读取图片并转换为RGB模式
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    
    #image = preprocess(image)                      # 应用预处理
    image = transform(image)
    


    return torch.from_numpy(image).unsqueeze(0)
    return image.unsqueeze(0)                      # 增加批量维度