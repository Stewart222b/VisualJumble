import torch.nn as nn
import torch.nn.functional as F

class MyLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2,2)
        #self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        #print(f'Conv1 output shape: \t{x.shape}')
        #x = self.sigmoid(x)
        x = F.sigmoid(x)
        x = self.pool1(x)
        #print(f'Average pool1 output shape: \t{x.shape}')

        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool2(x)
        #x = self.conv3(x)

        #print(f'Average pool2 output shape: \t{x.shape}')
        x = self.flatten(x)
        #print(f'Flatten output shape: \t{x.shape}')
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x