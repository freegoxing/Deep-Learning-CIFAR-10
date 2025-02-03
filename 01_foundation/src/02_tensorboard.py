from torch.utils.tensorboard import SummaryWriter
import random
from PIL import Image
import numpy as np

"""
tensorboard 可视化
1. 安装 pip install tensorboard
2. 先运行py文件在文件夹下有新出来的事件文件
3. 运行 tensorboard --logdir=logs
                   --port= 指定端口（默认6006）
4. 浏览器打开
"""

# 指定log文件路径
writer = SummaryWriter('../logs/02_tensorboard')

# img_path = "./Data/train/bees_image/16838648_415acd9e3f.jpg" # global_step =1
img_path = "../Data/train/ants_image/0013035.jpg"  # global_step =2
img = Image.open(img_path)
img_array = np.array(img)

writer.add_image(
    # 标题
    tag='test',
    # 图片
    img_tensor=img_array,
    # 步数
    global_step=2,
    # 数据格式
    dataformats='HWC'
)


for i in range(100):
    # 标签，y坐标， x坐标
    writer.add_scalar(
        # 标题
        tag='y=2x的一个模拟',
        # y坐标
        scalar_value=2*i+random.uniform(-2,2),
        # x坐标
        global_step=i
    )

writer.close()