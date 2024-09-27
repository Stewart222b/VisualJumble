import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import argparse
import sys

from utils import *
from .lenet import MyLeNet

def parse_args():
    '''
    Parse arguments | 解析参数
    '''
    # 创建解析器
    parser = argparse.ArgumentParser(description='Parse CLI arguments | 解析命令行参数')

    # 添加参数
    parser.add_argument('--device', type=str, default='cpu', help='Training device | 训练设备')
    parser.add_argument('--model', type=str, default='lenet', help='models/networks | 模型/网络')
    parser.add_argument('--epoch', type=int, default=5, help='Training epoch | 训练轮数')
    parser.add_argument('--batch', type=int, default=64, help='Training batch | 训练批次')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')

    # 解析参数
    args = parser.parse_args()
    return args

def main(args):
    if args.device == 'mps':
        # Check that MPS is available
        if not torch.backends.mps.is_available():
            device = torch.device('cpu')
            if not torch.backends.mps.is_built():
                print(Warning("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."))
            else:
                print(Warning("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."))

        else:
            device = torch.device('mps')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    print(Info(f'using device: {device}\n'))



    # 数据转换：将图像转换为张量并进行标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 加载训练和测试数据集
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    
    if args.model == 'lenet':
        model = MyLeNet()
    else:
        model = MyLeNet()

    model.to(device)


    # 训练过程
    losses = []
    accuracies = []

    lr = 0.2
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        running_loss = 0.0
        model.train()  # 设置为训练模式

        # 训练
        progress = Progress(bar_length=20, show_time=True, time_digit=1)
        for i, (images, labels) in enumerate(train_loader):
            info = f'Training epoch [{epoch+1}/{args.epoch}]: '
            progress.progress_bar(i+1, len(train_loader), info)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)

        # 验证
        model.eval()  # 设置为评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 不计算梯度
            progress = Progress(bar_length=20, show_time=True, time_digit=1)
            print()
            for i, (images, labels) in enumerate(test_loader):
                info = f'Evaluating:'
                progress.progress_bar(i+1, len(test_loader), info)
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        accuracies.append(accuracy)
        print(Info(f'Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy*100:.2f}\n\n'))


    # 可视化损失和准确率
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(range(1, args.epoch + 1), losses, 'g-', marker='o', label='Loss')
    ax2.plot(range(1, args.epoch + 1), accuracies, 'b-', marker='o', label='Validation Accuracy')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='g')
    ax2.set_ylabel('Validation Accuracy', color='b')
    ax1.set_xticks(range(1, args.epoch + 1))
    ax1.grid()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Training Loss and Validation Accuracy')
    plt.show()