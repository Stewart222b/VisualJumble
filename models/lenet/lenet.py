from model_info import ModelInfo
import torch.nn as nn
import torch.nn.functional as F

class MyLeNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(16, 26, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(26)
        self.pool3 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(26 * 3 * 3, 120)

        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #print(f'Conv1 output shape: \t{x.shape}')
        x = self.pool1(x)
        #print(f'Max pool1 output shape: \t{x.shape}')

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        #print(f'Max pool2 output shape: \t{x.shape}')

        x = F.relu(self.bn3(self.conv3(x)))
        #print(f'Conv3 output shape: \t{x.shape}')

        x = self.flatten(x)
        #print(f'Flatten output shape: \t{x.shape}')
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class LeNetInfo(ModelInfo):
    def __init__(self, **kwargs) -> None:
        input = (28, 28)
        super().__init__(input, **kwargs)
        
    def pre(self, img):    
        return img
    
    def post(self, output):
        return output