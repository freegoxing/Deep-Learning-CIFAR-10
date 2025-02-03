from PIL import Image
from torchvision import transforms
import torch
import cv2
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_path = "../Data/train/ants_image/0013035.jpg"

writer = SummaryWriter('../logs/03_transforms')

"""
ToTensor
"""

# 从python数据类型到tensor数据类型
# 1. 创建实例
# tensor = transforms.ToTensor()
# 2. 调用实例
# input = tensor(数据)

tensor = transforms.ToTensor()

# PIL 图片数据类型
img_PIL = Image.open(img_path)
tensor_img_PIL = tensor(img_PIL).to(device)
# print(tensor_img_PIL)


# opencv 图片数据类型
img_CV = cv2.imread(img_path)
tensor_img_CV = tensor(img_CV).to(device)
# print(tensor_img_CV)


# writer.add_image('tensor_img', tensor_img_PIL, global_step=0)
# writer.add_image('tensor_img', tensor_img_CV, global_step=1)


"""
Normalize
"""

# 归一化

norm = transforms.Normalize(
    mean=[6,5,3],
    std=[4,6,2]
)

img_norm = norm(tensor_img_PIL)
# writer.add_image('norm_img', img_norm, global_step=2)

"""
Resize
"""

# print(img_PIL.size)
resize = transforms.Resize((256,256))
img_resize = resize(img_PIL)
img_resize = tensor(img_resize).to(device)
# print(img_resize.shape)
# writer.add_image('resize_img(768x512 -> 256x256)',img_resize, global_step=0)

resize_2 = transforms.Resize(256)
img_resize_2 = resize_2(img_PIL)
img_resize_2 = tensor(img_resize_2)
# writer.add_image('resize_img(768x512->256x[])', img_resize_2, 0)

"""
Compose
"""

# 这个是把一系列操作合并为一个连贯的操作

compose = transforms.Compose([resize_2, tensor])
img_compose = compose(img_PIL)
# writer.add_image('compose_img', img_compose, 0)


"""
RandomCrop
"""

# 它用于从输入图像中随机裁剪出指定大小的子图像
# 如下面是从PIL类型的图片中随机裁剪出256x256的子图像
# 或输入RandomCrop((200, 300)) 是指裁剪出(200, 200)的子图像

random_crop = transforms.RandomCrop(256)
compose_2 = transforms.Compose([random_crop, tensor])
for i in range(10):
    img_crop = compose_2(img_PIL)
    # writer.add_image('random_crop_img', img_crop, i)

writer.close()
