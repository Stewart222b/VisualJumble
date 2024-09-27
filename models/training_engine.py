import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Parse CLI arguments | 解析命令行参数')

    # 添加参数
    parser.add_argument('--device', type=str, default='mps', help='Training device | 训练设备')
    parser.add_argument('--model', type=str, default='lenet', help='models/networks | 模型/网络')
    parser.add_argument('--epoch', type=int, default=10, help='Training epoch | 训练轮数')
    parser.add_argument('--batch', type=int, default=256, help='Training batch | 训练批次')
    parser.add_argument('--lr', type=float, default=0.2, help='Learning rate| 学习率')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', help=f'Available dataset:{TrainEngine.available_dataset}')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')

    # 解析参数
    args = parser.parse_args()
    return args

class TrainEngine:
    available_net = ['lenet']
    available_dataset = ['fashion_mnist']
    available_optim = ['sgd']
    available_criterion = ['ce', 'bce']

    def __init__(self,):
        args = parse_args()

        self.device = self.get_device(args.device)
        self.model = self.get_model(args.model)
        self.epoch = self.get_epoch(args.epoch)
        self.batch = self.get_batch(args.batch)
        self.lr = self.get_lr(args.lr)

        self.optimizer = self.get_optim()
        self.criterion = self.get_criterion()

        self.train_loader, self.val_loader = self.load_data(args.dataset)

        # 训练过程
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []


    def get_device(self, device: str):
        if device == 'mps':
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
        elif self.args.device == 'cpu':
            device = torch.device('cpu')
        else:
            device = torch.device('cpu')

        print(Info(f'Using device: {device}'))
        return device
    
    
    def get_model(self, model_name: str):
        if model_name == 'lenet':
            from .lenet import MyLeNet
            model = MyLeNet()
        else:
            #model = MyLeNet()
            print(Error(f'please enter a valid model or network: {TrainEngine.available_net}'))
            return -1

        model.to(self.device)
        return model
    
    
    def get_epoch(self, epoch: int):
        if epoch < 1:
            print(Error(f'Invalid epoch: {epoch}.'))
            return -1

        return epoch
    
    
    def get_batch(self, batch: int):
        if batch < -1 or batch == 0:
            print(Error(f'Invalid batch size: {batch}. i.e. batch -> (0, inf), -1 for auto-batch'))
            return -1
        
        if batch == -1:
            print(Warning('Using auto-batch!'))

        return batch
    
    
    def get_lr(self, lr: float):
        if lr <= 0 or lr >= 1:
            print(Error(f'Invalid learning rate: {lr}. i.e. lr -> (0, 1)'))
            return -1
        
        return lr

    
    def get_optim(self,):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)
    
    
    def get_criterion(self, ):
        return nn.CrossEntropyLoss()
    
    
    def load_data(self, dataset: str):
        # 数据转换：将图像转换为张量并进行标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 归一化
        ])
        # 加载训练和测试数据集
        if dataset == 'fashion_mnist':
            train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            val_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        else:
            print(Error(''))

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch, shuffle=False)

        return train_loader, val_loader
    
    
    def train(self,):
        self.model.train()  # 设置为训练模式
        total = 0
        running_loss = 0.0
        correct = 0
        progress = Progress(bar_length=20, show_time=True, time_digit=1)

        for i, (images, labels) in enumerate(self.train_loader):
            progress.progress_bar(i+1, len(self.train_loader))
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = correct / total * 100

        return train_loss, train_acc
    

    def eval(self,):
        self.model.eval()  # 设置为评估模式
        total = 0
        running_loss = 0.0  # 验证损失
        correct = 0
        progress = Progress(bar_length=20, show_time=True, time_digit=1)

        with torch.no_grad():  # 不计算梯度
            for i, (images, labels) in enumerate(self.val_loader):
                progress.progress_bar(i+1, len(self.val_loader))
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = correct / total * 100

        return val_loss, val_acc


    def visualize(self,):
        # 可视化损失和准确率
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(range(1, self.epoch + 1), self.train_losses, 'g-', marker='o', label='Train Loss')
        ax1.plot(range(1, self.epoch + 1), self.val_losses, 'r-', marker='x', label='Val Loss')  # 验证损失
        ax2.plot(range(1, self.epoch + 1), self.train_accuracies, 'm-', marker='x', label='Train Accuracy')
        ax2.plot(range(1, self.epoch + 1), self.val_accuracies, 'b-', marker='o', label='Validation Accuracy')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='g')
        ax2.set_ylabel('Accuracy', color='b')
        ax1.set_xticks(range(1, self.epoch + 1))
        ax1.grid()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.title('Loss and Accuracy')
        plt.show()


    def start(self):
        for epoch in range(self.epoch):
            print(Info(f'Training epoch | 训练轮数: [{epoch+1}/{self.epoch}]'))
            # Train 训练
            train_loss, train_acc = self.train()

            print(f'Evaluating: | 验证中：')
            # Val 验证
            val_loss, val_acc = self.eval()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(Info(f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.2f}\n'))

        self.visualize()