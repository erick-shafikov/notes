# Conv2d

[Сигнатура класса и нативная реализация](../../libs/pytorch/models/conv2d.md)

Пример использования

```python
import torch
import torch.nn as nn

C = 5  # число каналов
H, W = 32, 24  # размеры изображения: H - число строк; W - число столбцов
kernel_size = (5, 3)  # размер ядра по осям (H, W)
stride = (1, 1)  # шаг смещения ядра по осям (H, W)
padding = 0  # размер нулевой области вокруг изображения (число строк и столбцов с каждой стороны)

# числа от 0 до 255 формой (C, H, W)
x = torch.randint(0, 255, (C, H, W), dtype=torch.float32)

# здесь продолжайте программу
layer_nn = nn.Conv2d(
    in_channels=C,
    out_channels=1,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding
)
# ожидает всегда четырёхмерный тензор с размерностью (batch_size, channels, height, width)
t_out = layer_nn(x.unsqueeze(0))
```

# Работа с MaxPool2d

[Сигнатура класса и нативная реализация](../../libs/pytorch/models/maxPool.md)

```python
import torch
import torch.nn as nn

H, W = 32, 25
x = torch.randint(0, 255, (H, W), dtype=torch.float32)

# здесь продолжайте программу
pool = nn.MaxPool2d(kernel_size=(3, 2))
t_out = pool(x.unsqueeze(0))
```

# Пример сверточных

```python
import torch
import torch.nn as nn

batch_size = 32
data_img = torch.rand(batch_size, 3, 16, 16)
data_x = torch.rand(batch_size, 12)


# здесь продолжайте программу
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(in_features=12, out_features=64, bias=False),
            nn.Sigmoid(),
            nn.BatchNorm1d(64),
        )

        self.output = nn.Linear(in_features=576, out_features=10, bias=True)

    def forward(self, img, x):
        img = self.net1(img)
        x = self.net2(x)

        img_x = torch.cat((img, x), dim=1)

        return self.output(img_x)


model = Model()
model.eval()

predict = model(data_img, data_x)
```

# Пример с изображением

```python
import os
import json
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class SunDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        # инициализация файлов
        self.path = os.path.join(path, "train" if train else "test")
        # преобразования
        self.transform = transform
        # преобразования
        with open(os.path.join(self.path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = len(self.format)
        self.files = tuple(self.format.keys())
        self.targets = tuple(self.format.values())

    def __getitem__(self, item):
        path_file = os.path.join(self.path, self.files[item])
        img = Image.open(path_file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.targets[item], dtype=torch.float32)

    def __len__(self):
        return self.length


model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding='same'), # (b, 32, 256, 256)
    nn.ReLU(),
    nn.MaxPool2d(2), # (b, 32, 128, 128)
    nn.Conv2d(32, 8, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2), # (b, 8, 64, 64)
    nn.Conv2d(8, 4, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2), # (b, 4, 32, 32)
    nn.Flatten(),
    nn.Linear(4096, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

transforms = tfs.Compose(
    [
        tfs.ToImage(),
        tfs.ToDtype(torch.float32, scale=True)
    ]
)

d_train = SunDataset("dataset_reg", transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

optimizer = optim.Adam(
    params=model.parameters(),
    lr=0.001,
    weight_decay=0.001
)

loss_function = nn.MSELoss()
epochs = 5
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

st = model.state_dict()
torch.save(st, 'model_sun_2.tar')

d_test = SunDataset("dataset_reg", train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=50, shuffle=False)

# тестирование обученной НС
Q = 0
count = 0
model.eval()

test_tqdm = tqdm(test_data, leave=True)
for x_test, y_test in test_tqdm:
    with torch.no_grad():
        p = model(x_test)
        Q += loss_function(p, y_test).item()
        count += 1

Q /= count
print(Q)
```

Создание сверточной НС с помощью datasets

```python
import torch
import torch.utils.data as data
import torch.nn as nn
# загрузка dataset Digits на диск
from sklearn.datasets import load_digits, _global_model_state

digits = load_digits()

# Признаки и целевые значения
_global_var_data_x = torch.tensor(digits.data, dtype=torch.float32).view(-1, 1, 8, 8)
_global_var_target = torch.tensor(digits.target, dtype=torch.int64)

ds = data.TensorDataset(_global_var_data_x, _global_var_target)

model = nn.Sequential(
    nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        bias=True,
    ),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(
        kernel_size=(2, 2),
        stride=2
    ),
    nn.Conv2d(
        in_channels=32,
        out_channels=16,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        bias=True,
    ),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(
        kernel_size=(2, 2),
        stride=2
    ),
    nn.Flatten(),
    nn.Linear(
        in_features=64,
        out_features=10,
        bias=True,
    )
)

d_train = data.DataLoader(ds)
model.load_state_dict(_global_model_state)

model.eval()
Q = 0
for x, y in d_train:
    Q += (torch.argmax(model(x)) == y).float().item()

Q /= len(d_train)
```
