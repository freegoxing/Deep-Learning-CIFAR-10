"""
模型的保存
"""

import torch
from torch.nn import Module, Conv2d
from torchvision import models
import os

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

model_dirs = "../models/15_model_save"

# # 保存方法1，模型结构+模型参数
# torch.save(vgg16, os.path.join(model_dirs, "vgg16_method1.pth"))


# # 保存方式2，参数（官方推荐）
# torch.save(vgg16.state_dict(), os.path.join(model_dirs, "vgg16_method2.pth"))

"""
陷阱
"""

class Freedom(Module):
    def __init__(self):
        super(Freedom, self).__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


    def forward(self, x):
        x = self.conv1(x)
        return x


freedom = Freedom()

torch.save(freedom, os.path.join(model_dirs, "freedom_method1.pth"))













