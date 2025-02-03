"""
优化器
"""

import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, Module, CrossEntropyLoss
from torch.optim import SGD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

dataset = datasets.CIFAR10('../dataset', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Freedom(Module):
    def __init__(self):
        super(Freedom, self).__init__()
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
        x = self.model(x)
        return x


loss = CrossEntropyLoss()
freedom = Freedom().to(device)
optimizer = SGD(
    # 模型参数
    params=freedom.parameters(),
    # 学习率
    lr=0.001)

for epoch in range(100):
    running_loss = 0.0
    for data in dataloader:
        optimizer.zero_grad()
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        output = freedom(imgs)
        result = loss(output, targets)
        result.backward()
        optimizer.step()
        running_loss += result

    print(running_loss)






