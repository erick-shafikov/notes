# Многоклассовая классификация (цифры по 28x28 изображения)

Пример реализации НС для распознавания цифр c использованием tqdm

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


class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(self.path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = 0
        self.files = []
        self.targets = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

    def __getitem__(self, item):
        path_file, target = self.files[item]
        t = self.targets[target]
        img = Image.open(path_file)

        if self.transform:
            img = self.transform(img).ravel().float() / 255.0

        return img, t

    def __len__(self):
        return self.length


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

to_tensor = tfs.ToImage()  # PILToTensor
d_train = DigitDataset("dataset", transform=to_tensor)
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

d_test = DigitDataset("dataset", train=False, transform=to_tensor)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)

Q = 0

# тестирование обученной НС
model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        y = torch.argmax(y_test, dim=1)
        Q += torch.sum(p == y).item()

Q /= len(d_test)
print(Q)
```

# Многоклассовая классификация (CrossEntropyLoss)

```python
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
import torch

iris = load_iris()
_global_var_data_x = torch.tensor(iris.data, dtype=torch.float32)
_global_var_target = torch.tensor(iris.target, dtype=torch.float32)


class IrisDataset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x  # тензор размерностью (150, 4), тип float32
        self.target = _global_var_target  # тензор размерностью (150, ), тип int64 (long)

        self.length = len(self.data)  # размер выборки
        self.categories = ['setosa' 'versicolor' 'virginica']  # названия классов
        self.features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    def __getitem__(self, item):
        return self.data[item], self.target[
            item]  # возврат образа по индексу item в виде кортежа: (данные, целевое значение)

    def __len__(self):
        return self.length  # возврат размера выборки


class IrisClassModel(nn.Module):
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        # модель нейронной сети из двух полносвязных слоев:
        # 1-й слой: число входов in_features, число нейронов 16
        # 2-й слой: число нейронов out_features
        self.layer1 = nn.Linear(in_features, 16)
        self.layer2 = nn.Linear(16, out_features)

    def forward(self, x):
        # тензор x пропускается через 1-й слой
        x = self.layer1(x)
        # через функцию активации torch.relu()
        x = torch.relu(x)
        # через второй слой
        x = self.layer2(x)
        # полученный (вычисленный) тензор возвращается
        return x


torch.manual_seed(11)

# создать модель IrisClassModel с числом входов 4 и числом выходов 3
model = IrisClassModel(4, 3)
# перевести модель в режим обучения
model.train()

epochs = 10  # число эпох обучения
batch_size = 8  # размер батча

# создать объект класса IrisDataset
d_train = IrisDataset()  # создать объект класса DataLoader с размером пакетов batch_size и перемешиванием образов выборки
train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True)

# создать оптимизатор Adam для обучения модели с шагом обучения 0.01
optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()  # создать функцию потерь с помощью класса CrossEntropyLoss (используется при многоклассовой классификации)

for _e in range(epochs):  # итерации по эпохам
    for x_train, y_train in train_data:
        predict = model(x_train)  # вычислить прогноз модели для данных x_train
        loss = loss_func(predict, y_train)  # вычислить значение функции потерь

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# перевести модель в режим эксплуатации
model.eval()
# выполнить прогноз модели по всем данным выборки


predict = model(d_train.data)
# вычислить долю верных классификаций (сохранить, как вещественное число, а не тензор)
Q = (torch.argmax(predict, dim=1) == d_train.target).float().mean().item()

```

# Многоклассовая классификация (повтор)

```python
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris

iris = load_iris()
_global_var_data_x = torch.tensor(iris.data, dtype=torch.float32)
_global_var_target = torch.tensor(iris.target, dtype=torch.float32)


class IrisDataset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x  # тензор размерностью (150, 4), тип float32
        self.target = _global_var_target  # тензор размерностью (150, ), тип int64 (long)

        self.length = len(self.data)  # размер выборки
        self.categories = ['setosa' 'versicolor' 'virginica']  # названия классов
        self.features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    def __getitem__(self, item):
        return self.data[item], self.target[
            item]  # возврат образа по индексу item в виде кортежа: (данные, целевое значение)

    def __len__(self):
        return self.length  # возврат размера выборки


class IrisClassModel(nn.Module):
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        # модель нейронной сети из двух полносвязных слоев:
        # 1-й слой: число входов in_features, число нейронов 16
        # 2-й слой: число нейронов out_features
        self.layer1 = nn.Linear(in_features, 16)
        self.layer2 = nn.Linear(16, out_features)

    def forward(self, x):
        # тензор x пропускается через 1-й слой
        x = self.layer1(x)
        # через функцию активации torch.relu()
        x = torch.relu(x)
        # через второй слой
        x = self.layer2(x)
        # полученный (вычисленный) тензор возвращается
        return x


torch.manual_seed(11)

# создать модель IrisClassModel с числом входов 4 и числом выходов 3
model = IrisClassModel(4, 3)
# перевести модель в режим обучения
model.train()

epochs = 10  # число эпох обучения
batch_size = 8  # размер батча

# создать объект класса IrisDataset
d_train = IrisDataset()  # создать объект класса DataLoader с размером пакетов batch_size и перемешиванием образов выборки
train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True)

# создать оптимизатор Adam для обучения модели с шагом обучения 0.01
optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()  # создать функцию потерь с помощью класса CrossEntropyLoss (используется при многоклассовой классификации)

for _e in range(epochs):  # итерации по эпохам
    for x_train, y_train in train_data:
        predict = model(x_train)  # вычислить прогноз модели для данных x_train
        loss = loss_func(predict, y_train)  # вычислить значение функции потерь

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# перевести модель в режим эксплуатации
model.eval()
# выполнить прогноз модели по всем данным выборки


predict = model(d_train.data)
# вычислить долю верных классификаций (сохранить, как вещественное число, а не тензор)
Q = (torch.argmax(predict, dim=1) == d_train.target).float().mean().item()

```

- копия предыдущего

```python
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine

wine_data = load_wine()

# Признаки и целевые значения
_global_var_data_x = torch.tensor(wine_data.data, dtype=torch.float32)
_global_var_target = torch.tensor(wine_data.target, dtype=torch.int64)


class WineDataset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x  # тензор размерностью (178, 13), тип float32
        self.target = _global_var_target  # тензор размерностью (178, ), тип int64 (long)

        self.length = len(self.data)  # размер выборки
        self.categories = ['class_0', 'class_1', 'class_2']  # названия классов

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return self.length  # возврат размера выборки


# здесь продолжайте программу
class WineClassModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(13, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        return self.layer3(x)


model = WineClassModel()
model.train()

epoches = 20
batch_size = 16

d_train = WineDataset()
train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True)
optimizer = optim.Adam(params=model.parameters(), lr=0.01)

loss_func = nn.CrossEntropyLoss()

for _ in range(epoches):
    for x_train, y_train in train_data:
        predict = model(x_train)
        loss = loss_func(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

predict = model(d_train.data)
Q = (torch.argmax(predict, dim=1) == d_train.target).float().mean().item()

```

- копия предыдущего

```python
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits

digits = load_digits()

# Признаки и целевые значения
_global_var_data_x = torch.tensor(digits.data, dtype=torch.float32)
_global_var_target = torch.tensor(digits.target, dtype=torch.int64)


# здесь продолжайте программу

class DigitDataset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x
        self.target = _global_var_target
        self.length = len(_global_var_data_x)

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return self.length


class DigitClassModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(64, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        return self.layer3(x)


model = DigitClassModel()
model.train()

d_train = DigitDataset()
batch_size = 12
train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True)

lr = 0.01
optimizer = optim.Adam(params=model.parameters(), lr=lr)

epoches = 10
loss_func = nn.CrossEntropyLoss()

for _ in range(epoches):
    for x_train, y_train in train_data:
        predict = model(x_train)
        loss = loss_func(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

predict = model(d_train.data)
Q = (torch.argmax(predict, dim=1) == d_train.target).float().mean().item()

```