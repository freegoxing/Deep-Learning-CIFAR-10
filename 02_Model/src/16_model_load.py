"""
模型的加载
"""

import torch
from torchvision import models
from torch.nn import Module, Conv2d
import os

# 模块不能以数字开头
# from 15_model_save import *


model_dirs = "../models/15_model_save"

# # 方式一 结构和参数
# vgg16 = torch.load(os.path.join(model_dirs, "vgg16_method1.pth"))
# print(vgg16)


# # 方式二 参数
# vgg16 = models.vgg16(weights=None)
# vgg16.load_state_dict(torch.load(os.path.join(model_dirs, "vgg16_method2.pth"), weights_only=True))
# print(vgg16)

"""
陷阱
"""

# 要把你在模型类中定义的网络结构写出来
# class Freedom(Module):
#     def __init__(self):
#         super(Freedom, self).__init__()
#         self.conv1 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# 或import 定义模型的文件

model = torch.load(os.path.join(model_dirs, "freedom_method1.pth"), weights_only=False)
print(model)
