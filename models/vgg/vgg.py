from models.model_info import ModelInfo
import torch
import torch.nn as nn

from utils.io import *

class MyVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(True))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2)
    

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class MyVGG(nn.Module):
    versions = { # int: number of kernels of conv, 'M': Max pooling layer | 数字代表卷积层的输出通道数，'M' 代表最大汇聚层
        'vgg11nano': [(1, 16), (1, 32), (2, 64), (2, 128), (2, 128)],
        # VGG 11: 8 convs, 3 fc | 8 个卷积层，3 个全连接层
        'vgg11': [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)],
        # VGG 13: 10 convs, 3 fc | 10 个卷积层，3 个全连接层
        'vgg13': [(2, 64), (2, 128), (2, 256), (2, 512), (2, 512)],
        # VGG 16: 13 convs, 3 fc | 13 个卷积层，3 个全连接层
        'vgg16': [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)],
        # VGG 19: 16 convs, 3 fc | 16 个卷积层，3 个全连接层
        'vgg19': [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)],
    }

    def __init__(self, version='vgg16', cls_num=10):
        super().__init__()

        self.layers = self.make_layers(version)
        
        out_channels = 512
        if version == 'vgg11nano':
            out_channels = 128

        self.fc = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, cls_num)
        )


    def make_layers(self, version):
        layers = []
        in_channels = 1

        if version in MyVGG.versions.keys():
            config = MyVGG.versions[version]
        else:
            print(Error("Version must be 'vgg11', 'vgg13', 'vgg16', or 'vgg19'."))
            exit()

        for num_convs, out_channels in config:
            layers.append(MyVGGBlock(in_channels, out_channels, num_convs))
            in_channels = out_channels

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class VGGInfo(ModelInfo):
    def __init__(self, **kwargs) -> None:
        input = (224, 224)
        super().__init__(input, **kwargs)
        
    def pre(self, img):
        return img
    
    def post(self, output):
        return output