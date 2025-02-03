import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, Module

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


if __name__ == '__main__':
    freedom = Freedom()
    input = torch.rand(32, 3, 32, 32)
    output = freedom(input)
    print(output.size())