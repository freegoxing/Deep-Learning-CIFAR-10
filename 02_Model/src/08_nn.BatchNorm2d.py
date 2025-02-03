"""
正则层（可加快训练速度）
"""

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import BatchNorm2d
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = datasets.CIFAR10('../dataset', train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Freedom(nn.Module):
    def __init__(self):
        super(Freedom, self).__init__()
        self.batchnorm1 = BatchNorm2d(
            # C from an expected input of size (N,C,H,W)
            num_features=3,
            # 是否自动计算输入数据的均值和方差
            affine=True,
            # 学习率
            eps=1e-05)


    def forward(self, x):
        output = self.batchnorm1(x)
        return output


freedom = Freedom().to(device)
writer = SummaryWriter('../logs/08_nn.BatchNorm2d')

for i, data in enumerate(dataloader):
    imgs, targets = data
    imgs = imgs.to(device)
    writer.add_images('input', imgs, i)
    output = freedom(imgs)
    writer.add_images('output', output, i)

writer.close()




