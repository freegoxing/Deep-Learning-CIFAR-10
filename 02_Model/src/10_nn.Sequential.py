"""
用Sequential来搭建针对CIFAR-10神经网络
"""

import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, Module
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = datasets.CIFAR10('../dataset', train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Freedom(Module):
    def __init__(self):
        super(Freedom, self).__init__()
        # 一般写法
        # self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        # self.maxpool1 = MaxPool2d(kernel_size=2)
        # self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        # self.maxpool2 = MaxPool2d(kernel_size=2)
        # self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        # self.maxpool3 = MaxPool2d(kernel_size=2)
        # self.flatten = Flatten()
        # self.liner1 = Linear(in_features=1024, out_features=64)
        # self.liner2 = Linear(in_features=64, out_features=10)

        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )


    def forward(self, x):
        # 一般写法
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.liner1(x)
        # x = self.liner2(x)

        # Sequential写法
        x = self.model(x)

        return x


freedom = Freedom().to(device)

writer = SummaryWriter('../logs/10_nn.Sequential')
# 获取一个批次的数据
batch = next(iter(dataloader))
# 假设批次数据的格式是 (inputs, labels)
inputs, _ = batch

writer.add_graph(freedom, inputs.to(device))

writer.close()



