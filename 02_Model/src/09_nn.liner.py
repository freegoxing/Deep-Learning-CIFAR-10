"""
线性层
"""

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = datasets.CIFAR10('../dataset', train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Freedom(nn.Module):
    def __init__(self):
        super(Freedom, self).__init__()
        self.liner1 = Linear(
            # C from an expected input of size (N,C,H,W)
            in_features=196608,
            # 是否自动计算输入数据的均值和方差
            out_features=10,
            # 学习率
            bias=True)


    def forward(self, x):
        output = self.liner1(x)
        return output


freedom = Freedom().to(device)

for i, data in enumerate(dataloader):
    imgs, targets = data
    imgs = imgs.to(device)

    # imgs = torch.reshape(imgs, (1, 1, 1, -1))
    # print(imgs.shape) -> torch.Size([64, 3, 32, 32])
    # print(imgs_resize.shape) -> torch.Size([1, 1, 1, 196608])

    # 把数据展平
    imgs = torch.flatten(imgs)
    outputs = freedom(imgs)




