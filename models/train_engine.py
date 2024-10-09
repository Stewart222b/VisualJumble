import argparse
import matplotlib.pyplot as plt
from typing import *
import torch
from torch.optim import *
from utils.io import *
from utils.config import *

def parse_args():
    parser = argparse.ArgumentParser(description='Training parameters| 训练参数')

    # 添加参数
    parser.add_argument('--device', type=str, default='mps', help='Training device | 训练设备')
    parser.add_argument('--model', type=str, default='lenet', help=f'Models/networks | 模型/网络: {MODELS}')
    parser.add_argument('--epoch', type=int, default=10, help='Training epoch | 训练轮数')
    parser.add_argument('--batch', type=int, default=256, help='Training batch | 训练批次')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate| 学习率')
    parser.add_argument('--optimizer', type=str, default='sgd', help=f'Optimizer | 优化器: {OPTIMIZERS}')
    parser.add_argument('--criterion', type=str, default='ce', help=f'Criterion | 损失函数: {CRITERIONS}')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', help=f'Available dataset: {DATASETS}')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')

    # 解析参数
    args = parser.parse_args()
    return args


class TrainEngine:
    def __init__(self, **kwargs):
        '''
        '''
        args = parse_args()

        self.device = get_device(args.device, kwargs.get('device', None))
        self.model = get_model(args.model, kwargs.get('model', None)).to(self.device)
        self.epoch = get_epoch(args.epoch, kwargs.get('epoch', None))
        self.batch = get_batch(args.batch, kwargs.get('batch', None))
        self.lr = get_lr(args.lr, kwargs.get('lr', None))
        self.optimizer = get_optim(args.optimizer, self.model, self.lr, kwargs.get('optimizer', None))
        self.criterion = get_criterion(args.criterion, kwargs.get('criterion', None))
        self.train_loader, self.val_loader = load_dataset(args.dataset, self.batch, kwargs.get('dataset', None))

        # 训练过程
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    
    def train(self,) -> Tuple[float, float]:
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
    

    def eval(self,) -> Tuple[float, float]:
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


    def visualize(self,) -> None:
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


    def start(self) -> None:
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
    
    def _print_output_shape(self, name):
        def inner_hook(module, input, output):
            print(f'Output shape of [{name}]: {tuple(output.shape)}')

        return inner_hook

    def get_output_shape(self, layer=None):
        input_tensor = torch.randn(1, 1, 224, 224).to(self.device)
        #input_tensor = torch.randn(1, 1, 28, 28).to(self.device)

        # 注册 hook
        hooks = []
        for name, module in self.model.named_modules():
            #print(module)
            #print(name)
            #print('\n---------------------\n')
            if module == layer:
                hook = module.register_forward_hook(self._print_output_shape(name))
                hooks.append(hook)
                break
            if name:
                hook = module.register_forward_hook(self._print_output_shape(name))
                hooks.append(hook)

        # 前向传播
        _ = self.model(input_tensor)

        # 移除所有 hooks
        for hook in hooks:
            hook.remove()