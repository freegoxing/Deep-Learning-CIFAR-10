"""
损失函数
"""

import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

"""
L1loss
"""

loss_L1 = L1Loss(
    # 计算loss的方式
    reduction='sum',
)
print(loss_L1(input, target))


"""
MSELoss
"""

loss_MSE = MSELoss()
print(loss_MSE(input, target))


"""
CrossEntropyLoss
"""

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = CrossEntropyLoss()

result = loss_cross(x, y)
print(result)



