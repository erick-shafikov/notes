Пример собственного преобразователя

```python
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import ImageFolder


# собственный преобразователь
class RavelTransform(nn.Module):
    def forward(self, item):
        return item.ravel()


class DigitNN(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x


model = DigitNN(28 * 28, 32, 10)

# трансформация каждого изображения
transforms = tfs.Compose(
    [
        tfs.ToImage(), # перевод в тензор
        tfs.Grayscale(),
        tfs.ToDtype(torch.float32, scale=True), # scale => [0, 1]
        RavelTransform(),
    ]
)
# замена DataFolder-а
d_train = ImageFolder("dataset/train", transform=transforms) # вернет [3,x,x] для 3 цветовых каналов
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()
epochs = 2
model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

d_test = ImageFolder("dataset/test", transform=transforms)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)

Q = 0

# тестирование обученной НС
model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        Q += torch.sum(p == y_test).item()

Q /= len(d_test)
print(Q)
```

# Трансформация изображений

```python
import torch
import torch.nn as nn
import torchvision.transforms as tfs


# import torchvision.transforms.v2 as tfs_v2 - недоступен на Stepik

# здесь объявляйте класс ToDtypeV1
class ToDtypeV1(nn.Module):
    def __init__(self, dtype, scale):
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def forward(self, item):
        item = item.to(dtype=self.dtype)
        return ((item - item.min()) / (item.max() - item.min())) if self.scale and self.dtype in (
            torch.float16, torch.float32, torch.float64) else item


H, W = 128, 128
img_orig = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)  # тензор в программе не менять

img_mean = img_orig.float().mean(dim=(1, 2))  # средние для каждого цветового канала (первая ось)
img_std = img_orig.float().flatten(1, 2).std(dim=1)  # стандартное отклонение для каждого цветового канала (первая ось)

# здесь продолжайте программу
transforms = tfs.Compose(
    [
        ToDtypeV1(dtype=torch.float32, scale=False),
        tfs.Normalize(mean=img_mean, std=img_std)
    ]
)

img = transforms(img_orig.float())

```

Добавление шума

```python
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs


# import torchvision.transforms.v2 as tfs_v2 - недоступен на Stepik

# здесь объявляйте класс AddNoise
class AddNoise(nn.Module):
    def __init__(self, volume):
        super().__init__()

        self.volume = volume

    def forward(self, item):
        item += torch.randn_like(item, dtype=torch.float32) * self.volume
        return item


img_pil = Image.new(mode="RGB", size=(128, 128), color=(0, 128, 255))

# здесь продолжайте программу
transforms = tfs.Compose([tfs.ToTensor(), AddNoise(volume=0.1)])

img = transforms(img_pil)

```
