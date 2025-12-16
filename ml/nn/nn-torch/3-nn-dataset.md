# Аппроксимирование функции

Аппроксимирование на математической функции и однослойной НС

```python
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

import torch


def func(x):
    return torch.sin(2 * x) + 0.2 * torch.cos(10 * x) + 0.1 * x ** 2


class FuncDataset(data.Dataset):
    def __init__(self):
        _x = torch.arange(-5, 5, 0.1)
        self.data = _x
        self.target = func(_x)  # значения функции в точках _x
        self.length = len(_x)  # размер обучающей выборки

    def __getitem__(self, item):
        # возврат образа по индексу item в виде кортежа: (данные, целевое значение)
        return self.data[item], self.target[item]

    def __len__(self):
        # возврат размера выборки
        return self.length


class FuncModel(nn.Module):
    def __init__(self):
        super().__init__()
        # модель однослойной полносвязной нейронной сети:
        # 1-й слой: число входов 3 (x, x^2, x^3), число нейронов 1
        self.layer1 = nn.Linear(3, 1)

    def forward(self, x):
        xx = torch.empty(x.size(0), 3)
        xx[:, 0] = x
        xx[:, 1] = x ** 2
        xx[:, 2] = x ** 3
        y = self.layer1(xx)
        return y


torch.manual_seed(1)

# создать модель FuncModel
# перевести модель в режим обучения
model = FuncModel()

epochs = 20  # число эпох обучения
batch_size = 8  # размер батча

d_train = FuncDataset()  # создать объект класса FuncDataset
train_data = data.DataLoader(
    d_train,
    batch_size,
    shuffle=True
)  # создать объект класса DataLoader с размером пакетов batch_size и перемешиванием образов выборки

optimizer = optim.Adam(
    params=model.parameters(),
    lr=0.01
)  # создать оптимизатор Adam для обучения модели с шагом обучения 0.01
loss_func = torch.nn.MSELoss()  # создать функцию потерь с помощью класса MSELoss

for _e in range(epochs):  # итерации по эпохам
    for x_train, y_train in train_data:
        predict = model(x_train).flatten()  # вычислить прогноз модели для данных x_train
        loss = loss_func(predict, y_train)  # вычислить значение функции потерь

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# перевести модель в режим эксплуатации
# выполнить прогноз модели по всем данным выборки (d_train.data)
model.eval()

predict = model(torch.arange(-5, 5, 0.1)).flatten()
y_train = func(torch.arange(-5, 5, 0.1))

# вычислить потери с помощью loss_func по всем данным выборки; значение Q сохранить в виде вещественного числа
Q = loss_func(predict, y_train)

```

# Аппроксимирование функции с двумя переменными

```python
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class FuncDataset(data.Dataset):
    def __init__(self):
        _range = torch.arange(-3, 3, 0.1)
        self.data = torch.tensor([(_x, _y) for _x in _range for _y in _range])
        self.target = self._func(self.data)
        self.length = len(self.data)  # размер обучающей выборки

    @staticmethod
    def _func(coord):
        _x, _y = coord[:, 0], coord[:, 1]
        return torch.sin(2 * _x) * torch.cos(3 * _y) + 0.2 * torch.cos(10 * _x) * torch.sin(
            8 * _x) + 0.1 * _x ** 2 + 0.1 * _y ** 2

    def __getitem__(self, item):
        return self.data[item], self.target[
            item]  # возврат образа по индексу item в виде кортежа: (данные, целевое значение)

    def __len__(self):
        return self.length  # возврат размера выборки


class FuncModel(nn.Module):
    def __init__(self):
        super().__init__()
        # модель однослойной полносвязной нейронной сети:
        # 1-й слой: число входов 6 (x, x^2, x^3, y, y^2, y^3), число нейронов 1
        self.layer1 = nn.Linear(6, 1)

    def forward(self, coord):
        x, y = coord[:, 0], coord[:, 1]
        x.unsqueeze_(-1)
        y.unsqueeze_(-1)

        xx = torch.empty(coord.size(0), 6)
        xx[:, :3] = torch.cat([x, x ** 2, x ** 3], dim=1)
        xx[:, 3:] = torch.cat([y, y ** 2, y ** 3], dim=1)
        y = self.layer1(xx)
        return y


# здесь продолжайте программу
model = FuncModel()

epochs = 20
batch_size = 16
d_train = FuncDataset()

train_data = data.DataLoader(
    d_train,
    batch_size,
    shuffle=True
)

optimizer = optim.RMSprop(
    params=model.parameters(),
    lr=0.01
)

loss_func = torch.nn.MSELoss()

for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train).flatten()
        loss = loss_func(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

predict = model(d_train.data).flatten()
Q = torch.mean((predict - d_train.target) ** 2)

```

# Прогноз по выборке

```python

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

_global_var_data_x = torch.tensor(diabetes.data, dtype=torch.float32)
_global_var_target = torch.tensor(diabetes.target, dtype=torch.float32)


class FuncDataset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x
        self.target = _global_var_target
        self.length = len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return self.length  # возврат размера выборки


class FuncModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 64)
        self.layer2 = nn.Linear(64, 1)

    def forward(self, coord):
        x = self.layer1(coord)
        x = torch.tanh(x)
        x = self.layer2(x)
        return x


# здесь продолжайте программу
model = FuncModel()

epochs = 10
batch_size = 8
d_train = FuncDataset()

train_data = data.DataLoader(
    d_train,
    batch_size,
    shuffle=True,
)

optimizer = optim.RMSprop(
    params=model.parameters(),
    lr=0.01
)

loss_func = torch.nn.MSELoss()

for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train).flatten()
        loss = loss_func(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

predict = model(d_train.data).flatten()
Q = torch.mean((predict - d_train.target) ** 2)
```
