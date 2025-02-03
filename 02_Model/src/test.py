from PIL import Image
from torchvision import transforms
from model import *

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

img_path = "../imgs/img.png"
image = Image.open(img_path).convert("RGB")

img = transform(image).to(device)
print(img.shape)

# 检查发现在之前第36次效果最好
torch.serialization.add_safe_globals([Freedom, set, Sequential, Conv2d, MaxPool2d, Flatten, Linear])  # 允许 Freedom 类
model = torch.load("../models/train/freedom_36.pth", weights_only=True).to(device)
# print(model)

img = torch.reshape(img, (1, 3, 32, 32))

model.eval()
with torch.no_grad():
    output = model(img)
print(output)

index = output.argmax(dim=1, keepdim=True)
print(target[index])










