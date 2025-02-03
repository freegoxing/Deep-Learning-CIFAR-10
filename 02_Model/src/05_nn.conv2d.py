"""
卷积层
"""
import torch
from torch import nn
from torch.nn import Conv2d
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = datasets.CIFAR10(root='../dataset', train=False, transform=transforms.ToTensor(), download=True)

dataloder = DataLoader(dataset, batch_size=64, shuffle=True)

class Freedom(nn.Module):
    def __init__(self):
        super(Freedom, self).__init__()
        self.conv1 = Conv2d(
            # 输入通道数
            in_channels=3,
            # 输出通道数
            out_channels=6,
            # 卷积核大小
            kernel_size=3,
            # 步长
            stride=1,
            # 填充
            padding=0,
        )


    def forward(self, x):
        x = self.conv1(x)
        return x


freedom = Freedom()
freedom.to(device)

writer = SummaryWriter("../logs/05_nn.conv.2d")

for i, data in enumerate(dataloder):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    output = freedom(inputs)
    print(inputs.shape) #-> torch.Size([64, 3, 32, 32])
    print(output.shape) #-> torch.Size([64, 6, 30, 30])


    writer.add_images("input", inputs, i)

    # 输出通道数为6，所以要将其转换为3通道
    # 第一个用-1 表示自动计算，第二个用3表示通道数，第三个和第四个用510表示宽和高
    output = torch.reshape(output, (-1, 3, 30, 30))

    writer.add_images("output", output, i)

writer.close()





