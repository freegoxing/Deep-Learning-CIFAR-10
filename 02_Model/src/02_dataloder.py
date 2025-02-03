from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

test_set = datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=dataset_transform,
    download=True
)

test_loder = DataLoader(
    # 要处理的数据集
    dataset=test_set,
    # 每次处理的样本数
    batch_size=128,
    # 是否打乱数据
    shuffle=True,
    # 数据加载器的进程数
    num_workers=0,
    # 落单数据是否丢弃
    drop_last=True
)


writer = SummaryWriter("../logs/02_dataloder")

for i, data in enumerate(test_loder):
    imgs, targets = data
    imgs = imgs.to(device)
    writer.add_images("dataloder", imgs, i)

writer.close()










