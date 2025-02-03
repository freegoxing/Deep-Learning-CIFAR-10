"""
池化层
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataset = datasets.CIFAR10(root='../dataset', train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Freedom(nn.Module):
    def __init__(self):
        super(Freedom, self).__init__()
        self.maxpool1 = nn.MaxPool2d(
            # 尺寸大小
            kernel_size=3,
            # 步长（默认为kernel_size）
            stride=3,
            # 填充
            padding=0,
            # 是否对输入进行下采样（采样区域数据小于kernel_size时是否进行采样）
            ceil_mode=True)


    def forward(self, x):
        x = self.maxpool1(x)
        return x


freedom = Freedom().to(device)

writer = SummaryWriter("../logs/06_nn.MaxPool2d")

for i, data in enumerate(dataloader):
    inputs, targets = data
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = freedom(inputs)

    # print(imgs.shape) -> torch.Size([64, 3, 32, 32])
    # print(output.shape) -> torch.Size([64, 3, 11, 11])


    writer.add_images("input", inputs, i, dataformats='NCHW')
    writer.add_images("output", outputs, i, dataformats='NCHW')

writer.close()

