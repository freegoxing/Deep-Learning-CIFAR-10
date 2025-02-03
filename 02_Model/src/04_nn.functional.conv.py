import torch
from torch.nn import functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])


kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])


input = torch.reshape(input, [1, 1, 5, 5])
kernel = torch.reshape(kernel, [1, 1, 3, 3])

output = F.conv2d(
    # 输入数据 且size为4（batch， channel， height， width）
    input=input,
    # 卷积核 且size为4（out_channel， in_channel， height， width）
    weight=kernel,
    # stride 步长（height， width）
    stride=1)

print(output)


output2 = F.conv2d(input, kernel, stride=2)
print(output2)

output3 = F.conv2d(input, kernel, stride=1,
                   # 填充上下左右各填充1圈0
                   padding=1)

print(output3)