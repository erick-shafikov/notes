# Разбиение выборки

```python
import torch
import torch.utils.data as data


# создание dataset
class FuncDataset(data.Dataset):
    def __init__(self):
        self.data = torch.arange(-4, 4, 0.01)
        self.length = len(self.data)

    @staticmethod
    def func(x):
        return x ** 2 + 0.5 * x - torch.sin(5 * x)

    def __getitem__(self, item):
        data_item = self.data[item]
        return data_item, self.func(data_item)

    def __len__(self):
        return self.length


data_set = FuncDataset()
# разбиение data_set в отношении 80 к 20
d_train, d_val = data.random_split(data_set, [0.8, 0.2])
train_data = data.DataLoader(d_train, batch_size=16, shuffle=True)
train_data_val = data.DataLoader(d_val, batch_size=100, shuffle=False)
```

- бинарная с разбиением 50/30/20

```python
import torch
import torch.utils.data as data

data_x = [(5.3, 2.3), (5.7, 2.5), (4.0, 1.0), (5.6, 2.4), (4.5, 1.5), (5.4, 2.3), (4.8, 1.8), (4.5, 1.5), (5.1, 1.5),
          (6.1, 2.3), (5.1, 1.9), (4.0, 1.2), (5.2, 2.0), (3.9, 1.4), (4.2, 1.2), (4.7, 1.5), (4.8, 1.8), (3.6, 1.3),
          (4.6, 1.4), (4.5, 1.7), (3.0, 1.1), (4.3, 1.3), (4.5, 1.3), (5.5, 2.1), (3.5, 1.0), (5.6, 2.2), (4.2, 1.5),
          (5.8, 1.8), (5.5, 1.8), (5.7, 2.3), (6.4, 2.0), (5.0, 1.7), (6.7, 2.0), (4.0, 1.3), (4.4, 1.4), (4.5, 1.5),
          (5.6, 2.4), (5.8, 1.6), (4.6, 1.3), (4.1, 1.3), (5.1, 2.3), (5.2, 2.3), (5.6, 1.4), (5.1, 1.8), (4.9, 1.5),
          (6.7, 2.2), (4.4, 1.3), (3.9, 1.1), (6.3, 1.8), (6.0, 1.8), (4.5, 1.6), (6.6, 2.1), (4.1, 1.3), (4.5, 1.5),
          (6.1, 2.5), (4.1, 1.0), (4.4, 1.2), (5.4, 2.1), (5.0, 1.5), (5.0, 2.0), (4.9, 1.5), (5.9, 2.1), (4.3, 1.3),
          (4.0, 1.3), (4.9, 2.0), (4.9, 1.8), (4.0, 1.3), (5.5, 1.8), (3.7, 1.0), (6.9, 2.3), (5.7, 2.1), (5.3, 1.9),
          (4.4, 1.4), (5.6, 1.8), (3.3, 1.0), (4.8, 1.8), (6.0, 2.5), (5.9, 2.3), (4.9, 1.8), (3.3, 1.0), (3.9, 1.2),
          (5.6, 2.1), (5.8, 2.2), (3.8, 1.1), (3.5, 1.0), (4.5, 1.5), (5.1, 1.9), (4.7, 1.4), (5.1, 1.6), (5.1, 2.0),
          (4.8, 1.4), (5.0, 1.9), (5.1, 2.4), (4.6, 1.5), (6.1, 1.9), (4.7, 1.6), (4.7, 1.4), (4.7, 1.2), (4.2, 1.3),
          (4.2, 1.3)]

data_y = [1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
          -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1,
          -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1,
          1, -1, -1, -1, -1, -1]


class ClassDataset(data.Dataset):
    def __init__(self):
        self.data = torch.tensor(data_x)
        self.values = torch.tensor(data_y)
        self.length = len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.values[item]

    def __len__(self):
        return self.length


data_set = ClassDataset()
d_train, d_val, d_test = data.random_split(data_set, [0.5, 0.3, 0.2])
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True, drop_last=True)
train_data_val = data.DataLoader(d_val, batch_size=50, shuffle=False)
test_data = data.DataLoader(d_test, batch_size=20, shuffle=False)

```

- процесс обучения, сопоставление данных между выборками

```python
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class FuncModel(nn.Module):
    def __init__(self):
        super().__init__()
        # модель однослойной полносвязной нейронной сети:
        # 1-й слой: число входов 5 (x, x^2, x^3, x^4, x^5), число нейронов 1
        self.layer = nn.Linear(5, 1)

    def forward(self, x):
        x.unsqueeze_(-1)
        xx = torch.cat([x, x ** 2, x ** 3, x ** 4, x ** 5], dim=1)
        y = self.layer(xx)
        return y


torch.manual_seed(1)

model = FuncModel()  # создать модель FuncModel

epochs = 20  # число эпох обучения
batch_size = 16  # размер батча

# данные обучающей выборки (значения функции)
data_x = torch.arange(-5, 5, 0.05)  # тензоры data_x, data_y не менять
data_y = torch.sin(2 * data_x) - 0.3 * torch.cos(8 * data_x) + 0.1 * data_x ** 2

ds = data.TensorDataset(data_x, data_y)  # создание dataset
d_train, d_val = data.random_split(ds, [0.7, 0.3])  # разделить ds на две части в пропорции: 70% на 30%
train_data = data.DataLoader(d_train, batch_size=batch_size,
                             shuffle=True)  # создать объект класса DataLoader для d_train с размером пакетов batch_size и перемешиванием образов выборки
train_data_val = data.DataLoader(d_val,
                                 batch_size=batch_size)  # создать объект класса DataLoader для d_val с размером пакетов batch_size и без перемешивания образов выборки

optimizer = optim.RMSprop(params=model.parameters(),
                          lr=0.01)  # создать оптимизатор RMSprop для обучения модели с шагом обучения 0.01
loss_func = nn.MSELoss()  # создать функцию потерь с помощью класса MSELoss

loss_lst_val = []  # список значений потерь при валидации
loss_lst = []  # список значений потерь при обучении

for _e in range(epochs):
    # перевести модель в режим обучения
    model.train()
    loss_mean = 0  # вспомогательные переменные для вычисления среднего значения потерь при обучении
    lm_count = 0

    for x_train, y_train in train_data:
        predict = model(x_train).squeeze(1)  # вычислить прогноз модели для данных x_train
        loss = loss_func(predict, y_train)  # вычислить значение функции потерь

        # сделать один шаг градиентного спуска для корректировки параметров модели
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # вычисление среднего значения функции потерь по всей выборке
        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean

    # валидация модели
    # перевести модель в режим эксплуатации
    model.eval()
    Q_val = 0
    count_val = 0

    for x_val, y_val in train_data_val:
        with torch.no_grad():
            # для x_val, y_val вычислить потери с помощью функции loss_func
            p = model(x_val).squeeze(1)
            loss = loss_func(p, y_val)
            Q_val += loss.item()
            count_val += 1
    # сохранить средние потери, вычисленные по выборке валидации, в переменной Q_val
    Q_val /= count_val

    loss_lst.append(loss_mean)
    loss_lst_val.append(Q_val)

# перевести модель в режим эксплуатации
model.eval()
# выполнить прогноз модели по всем данным выборки (ds.data)
predict = model(data_x).squeeze(1)
# вычислить потери с помощью loss_func по всем данным выборки ds; значение Q сохранить в виде вещественного числа
Q = loss_func(data_y, predict).item()
```