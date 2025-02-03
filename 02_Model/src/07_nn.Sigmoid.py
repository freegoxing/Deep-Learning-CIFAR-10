"""
非线性激活层
"""

import torch
from torch import nn
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = datasets.CIFAR10(root='../dataset', train=False, transform=transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Freedom(nn.Module):
    def __init__(self):
        super(Freedom, self).__init__()
        # inplace 表示是否直接修改输入
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        output = self.sigmoid1(x)
        return output


freedom = Freedom().to(device)

writer = SummaryWriter('../logs/07_nn.Sigmoid')

for i, data in enumerate(dataloader):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = freedom(inputs)
    writer.add_images("inputs", inputs, i)
    writer.add_images("outputs", outputs, i)


writer.close()





