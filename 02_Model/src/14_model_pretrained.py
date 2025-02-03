"""
使用预训练模型
"""

import torch
import torchvision.models as models


# 下载预训炼的VGG16模型
pretrained_vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
non_pretrained_vgg16 = models.vgg16(weights=None)

# 添加自定义层

pretrained_vgg16.add_module("NEW_fc", torch.nn.Linear(1000, 10))
print(pretrained_vgg16)

# 或在内层添加

pretrained_vgg16.classifier.add_module("NEW_fc_inner", torch.nn.Linear(1000, 1000))
print(pretrained_vgg16)

# 修改内层

pretrained_vgg16.classifier[5] = torch.nn.Linear(1000, 100)
print(pretrained_vgg16)

