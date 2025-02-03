from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


dataset_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

train_set = datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = datasets.CIFAR10(root="./dataset",train=False, transform=dataset_transform, download=True)

# img, target = train_set[0]
# print(img)
# print(target)
# print(train_set.classes[target])
# img.show()

writer = SummaryWriter('../logs/01_dataset_torchvision')

for i in range(10):
    img, target = train_set[i]
    writer.add_image("train_set", img, i)


writer.close()


