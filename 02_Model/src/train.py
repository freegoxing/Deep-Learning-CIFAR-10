"""
完整的训练过程
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import *

# 设备准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 准备数据集
train_data = datasets.CIFAR10('../dataset', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.CIFAR10('../dataset', train=False, transform=transforms.ToTensor(), download=True)

print(f"训练数据集 {len(train_data)}")
print(f"测试数据集 {len(test_data)}")

train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
freedom = Freedom().to(device)

# 损失函数
loss_fn = CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 5e-3
optimizer = torch.optim.SGD(freedom.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 40

# 添加tensorboard
writer = SummaryWriter("../logs/train")

for epoch in range(epoch):
    print(f"-------------------第{epoch+1}轮训练开始-------------------")

    # 训练步骤开始
    freedom.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)


        optimizer.zero_grad()
        outputs = freedom.forward(imgs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_step += 1

        if total_train_step % 100 == 0:
            print(f"step{total_train_step}, train_loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        freedom.eval()
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = freedom.forward(imgs)

            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy


    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy / len(test_data)}")
    total_test_step += 1
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / len(test_data), total_test_step)

    torch.save(freedom, f"../models/train/freedom_{epoch+1}.pth")
    print(f"第{epoch+1}轮模型已保存")

torch.save(freedom, f"../models/train/freedom_final.pth")
writer.close()
