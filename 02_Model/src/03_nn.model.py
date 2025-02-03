"""
基本骨架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Freedom(nn.Module):
    def __init__(self):
        super(Freedom, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


freedom = Freedom()
x = torch.randn(1, 2, 2)
print(x)
output = freedom(x)
print(output)






