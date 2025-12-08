# Реализация нейронов

Для создания нейронов используется класс [Linear](../_libs/pytorch/models/linear.md)

Пример присвоения весов

```python
import torch
import torch.nn as nn

# тензор x в программе не менять
x = torch.tensor(list(map(float, input().split())), dtype=torch.float32)

# здесь продолжайте программу
layer = nn.Linear(16, 1, bias=False)
layer.weight.data = torch.ones_like(layer.weight.data)
print(f"{layer(x).item():.1f}")
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# создание слоев НС, инициализация конфигурации
layer1 = nn.Linear(in_features=3, out_features=2)
layer2 = nn.Linear(2, 1)


# сумматор + функция активации
# коэффициенты по умолчанию ([-1/sqrt(n), 1/sqrt(n)])
def forward(inp, l1: nn.Linear, l2: nn.Linear):
    u1 = l1.forward(inp)
    s1 = F.tanh(u1)

    u2 = l2.forward(s1)
    s2 = F.tanh(u2)
    return s2


# получить значения параметров
print(layer1.weight)
print(layer2.weight)
print(layer1.bias)
print(layer2.bias)

# передаем значения
# веса и баис первого нерона
layer1.weight.data = torch.tensor([[0.7402, 0.6008, -1.3340], [0.2098, 0.4537, -0.7692]])
layer1.bias.data = torch.tensor([0.5505, 0.3719])

# передаем значения
# веса и баис второго нерона
layer2.weight.data = torch.tensor([[-2.0719, -0.9485]])
layer2.bias.data = torch.tensor([-0.1461])

# входные значения
x = torch.FloatTensor([1, -1, 1])
# запуск функции
y = forward(x, layer1, layer2)
print(y.data)
```

# nn.Module

Краткое описание

```python
import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self, params):
        super().__init__()  # вызов инициализатора базового класса
        # создание и инициализация переменных модуля

    def forward(self, x):
        # реализация прямого прохода вектора x по нейронной сети
        return  # возврат тензора с выходными значениями нейронной сети
```

- двухслойная нс

```python
import torch
import torch.nn as nn


class TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.layer1 = nn.Linear(3, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        return x


x = torch.rand(3)

model = TwoLayerModel()

predict = model(x)  # predict = model.forward(x)

```

# С произвольным числом входов и выходов для 3 слоев

```python
import torch
import torch.nn as nn

batch_size = 4
x = torch.rand(batch_size, 32)


class ThreeLayersModel(nn.Module):
    def __init__(self, input_1, input_2, input_3, output):
        super().__init__()
        super().__init__()
        self.layer1 = nn.Linear(input_1, input_2)
        self.layer2 = nn.Linear(input_2, input_3)
        self.layer3 = nn.Linear(input_3, output)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        x = torch.tanh(x)
        return x


model = ThreeLayersModel(32, 10, 12, 1)

predict = model(x)
```

```python
import torch
import torch.nn as nn

x_train = torch.tensor(
    [(5.8, 1.2), (5.6, 1.5), (6.5, 1.5), (6.1, 1.3), (6.4, 1.3), (7.7, 2.0), (6.0, 1.8), (5.6, 1.3), (6.0, 1.6),
     (5.8, 1.9), (5.7, 2.0), (6.3, 1.5), (6.2, 1.8), (7.7, 2.3), (5.8, 1.2), (6.3, 1.8), (6.0, 1.0), (6.2, 1.3),
     (5.7, 1.3), (6.3, 1.9), (6.7, 2.5), (5.5, 1.2), (4.9, 1.0), (6.1, 1.4), (6.0, 1.6), (7.2, 2.5), (7.3, 1.8),
     (6.6, 1.4), (5.6, 2.0), (5.5, 1.0), (6.4, 2.2), (5.6, 1.3), (6.6, 1.3), (6.9, 2.1), (6.8, 2.1), (5.7, 1.3),
     (7.0, 1.4), (6.1, 1.4), (6.1, 1.8), (6.7, 1.7), (6.0, 1.5), (6.5, 1.8), (6.4, 1.5), (6.9, 1.5), (5.6, 1.3),
     (6.7, 1.4), (5.8, 1.9), (6.3, 1.3), (6.7, 2.1), (6.2, 2.3), (6.3, 2.4), (6.7, 1.8), (6.4, 2.3), (6.2, 1.5),
     (6.1, 1.4), (7.1, 2.1), (5.7, 1.0), (6.8, 1.4), (6.8, 2.3), (5.1, 1.1), (4.9, 1.7), (5.9, 1.8), (7.4, 1.9),
     (6.5, 2.0), (6.7, 1.5), (6.5, 2.0), (5.8, 1.0), (6.4, 2.1), (7.6, 2.1), (5.8, 2.4), (7.7, 2.2), (6.3, 1.5),
     (5.0, 1.0), (6.3, 1.6), (7.7, 2.3), (6.4, 1.9), (6.5, 2.2), (5.7, 1.2), (6.9, 2.3), (5.7, 1.3), (6.1, 1.2),
     (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), (7.9, 2.0), (5.6, 1.1), (7.2, 1.8), (5.5, 1.3), (7.2, 1.6), (6.3, 2.5),
     (6.3, 1.8), (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), (5.5, 1.3), (5.5, 1.1), (5.9, 1.5), (6.0, 1.5),
     (5.9, 1.8)])


class TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.layer1 = nn.Linear(2, 3, True)
        self.layer2 = nn.Linear(3, 1, True)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x


model = TwoLayerModel()

with torch.no_grad():
    predict = model(x_train)

```

# Примеры НС с обучением

- двухслойная НС с двумя слоями, с переменным количеством слоев

```python
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from numpy.random import randint


class NN(nn.Module):
    # input_dim - количество входных
    # num_hidden - количество выходных на первом слое == количество входных на втором
    # output_dim - количество выходных на второго
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        return x


model = NN(3, 2, 1)  # число входов 3; число нейронов 2 и 1
# пример
# model1 = NN(3, 5, 2)  # число входов 3; число нейронов 5 и 2
# model2 = NN(100, 18, 10)  # число входов 100; число нейронов 18 и 10

gen_p = model.parameters()  # возвращает генератор с набором параметров

# выборка
x_train = torch.FloatTensor([(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                             (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])
total = len(y_train)

# обучение модели

# оптимизаторы
optimizer = optim.RMSprop(params=model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()
# перевод модели в режим обучения
model.train()

# запуск SGD
for _ in range(1000):
    k = randint(0, total - 1)
    y = model(x_train[k])
    loss = loss_func(y, y_train[k])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# режим эксплуатации
model.eval()
model.requires_grad_(False)  # без менеджера контекста

# проверка по обучающей выборке
for x, d in zip(x_train, y_train):
    # оптимизация вычислений
    with torch.no_grad():
        # y - значение по модели
        # d - значение по факту
        y = model(x)
        print(f"Выходное значение НС: {y.data} => {d}")
```

- С вычислением качества прогноза

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# здесь объявляйте класс ClassModel
class ClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)
        self.layer2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


np.random.seed(1)
torch.manual_seed(1)

# обучающая выборка: x_train - входные значения; y_train - целевые значения
x_train = torch.tensor(
    [(5.8, 1.2), (5.6, 1.5), (6.5, 1.5), (6.1, 1.3), (6.4, 1.3), (7.7, 2.0), (6.0, 1.8), (5.6, 1.3), (6.0, 1.6),
     (5.8, 1.9), (5.7, 2.0), (6.3, 1.5), (6.2, 1.8), (7.7, 2.3), (5.8, 1.2), (6.3, 1.8), (6.0, 1.0), (6.2, 1.3),
     (5.7, 1.3), (6.3, 1.9), (6.7, 2.5), (5.5, 1.2), (4.9, 1.0), (6.1, 1.4), (6.0, 1.6), (7.2, 2.5), (7.3, 1.8),
     (6.6, 1.4), (5.6, 2.0), (5.5, 1.0), (6.4, 2.2), (5.6, 1.3), (6.6, 1.3), (6.9, 2.1), (6.8, 2.1), (5.7, 1.3),
     (7.0, 1.4), (6.1, 1.4), (6.1, 1.8), (6.7, 1.7), (6.0, 1.5), (6.5, 1.8), (6.4, 1.5), (6.9, 1.5), (5.6, 1.3),
     (6.7, 1.4), (5.8, 1.9), (6.3, 1.3), (6.7, 2.1), (6.2, 2.3), (6.3, 2.4), (6.7, 1.8), (6.4, 2.3), (6.2, 1.5),
     (6.1, 1.4), (7.1, 2.1), (5.7, 1.0), (6.8, 1.4), (6.8, 2.3), (5.1, 1.1), (4.9, 1.7), (5.9, 1.8), (7.4, 1.9),
     (6.5, 2.0), (6.7, 1.5), (6.5, 2.0), (5.8, 1.0), (6.4, 2.1), (7.6, 2.1), (5.8, 2.4), (7.7, 2.2), (6.3, 1.5),
     (5.0, 1.0), (6.3, 1.6), (7.7, 2.3), (6.4, 1.9), (6.5, 2.2), (5.7, 1.2), (6.9, 2.3), (5.7, 1.3), (6.1, 1.2),
     (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), (7.9, 2.0), (5.6, 1.1), (7.2, 1.8), (5.5, 1.3), (7.2, 1.6), (6.3, 2.5),
     (6.3, 1.8), (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), (5.5, 1.3), (5.5, 1.1), (5.9, 1.5), (6.0, 1.5),
     (5.9, 1.8)])
y_train = torch.FloatTensor(
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1,
     1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
     1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1])

model = ClassModel()  # здесь создавайте модель
# переведите модель в режим обучения

total = x_train.size(0)  # размер обучающей выборки
N = 1000  # число итераций алгоритма SGD

optimizer = optim.Adam(params=model.parameters(), lr=0.01)  # задайте оптимизатор Adam с шагом обучения lr=0.01
loss_func = torch.nn.BCEWithLogitsLoss()  # сформируйте функцию потерь (бинарную кросс-энтропию) с помощью класса nn.BCEWithLogitsLoss

for _ in range(N):
    k = np.random.randint(0, total)
    # пропустите через модель k-й образ выборки x_train и вычислите прогноз predict
    y = model(x_train[k])
    loss = loss_func(y[0], y_train[k])  # вычислите значение функции потерь и сохраните результат в переменной loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # выполните один шаг градиентного спуска так, как это было сделано в предыдущем подвиге

# переведите модель в режим эксплуатации
# прогоните через модель обучающую выборку и подсчитайте долю верных классификаций
# результат (долю верных классификаций) сохраните в переменной Q (в виде вещественного числа, а не тензора)
model.eval()

# оптимизация вычислений
with torch.no_grad():
    predict = (model(x_train).flatten() > 0).float()

Q = (y_train == predict).float().mean()
```

# Обучение по batch-ам

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# сюда скопируйте класс ClassModel из предыдущего подвига
class ClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)
        self.layer2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


np.random.seed(1)
torch.manual_seed(1)

# обучающая выборка: x_train - входные значения; y_train - целевые значения
x_train = torch.tensor(
    [(5.8, 1.2), (5.6, 1.5), (6.5, 1.5), (6.1, 1.3), (6.4, 1.3), (7.7, 2.0), (6.0, 1.8), (5.6, 1.3), (6.0, 1.6),
     (5.8, 1.9), (5.7, 2.0), (6.3, 1.5), (6.2, 1.8), (7.7, 2.3), (5.8, 1.2), (6.3, 1.8), (6.0, 1.0), (6.2, 1.3),
     (5.7, 1.3), (6.3, 1.9), (6.7, 2.5), (5.5, 1.2), (4.9, 1.0), (6.1, 1.4), (6.0, 1.6), (7.2, 2.5), (7.3, 1.8),
     (6.6, 1.4), (5.6, 2.0), (5.5, 1.0), (6.4, 2.2), (5.6, 1.3), (6.6, 1.3), (6.9, 2.1), (6.8, 2.1), (5.7, 1.3),
     (7.0, 1.4), (6.1, 1.4), (6.1, 1.8), (6.7, 1.7), (6.0, 1.5), (6.5, 1.8), (6.4, 1.5), (6.9, 1.5), (5.6, 1.3),
     (6.7, 1.4), (5.8, 1.9), (6.3, 1.3), (6.7, 2.1), (6.2, 2.3), (6.3, 2.4), (6.7, 1.8), (6.4, 2.3), (6.2, 1.5),
     (6.1, 1.4), (7.1, 2.1), (5.7, 1.0), (6.8, 1.4), (6.8, 2.3), (5.1, 1.1), (4.9, 1.7), (5.9, 1.8), (7.4, 1.9),
     (6.5, 2.0), (6.7, 1.5), (6.5, 2.0), (5.8, 1.0), (6.4, 2.1), (7.6, 2.1), (5.8, 2.4), (7.7, 2.2), (6.3, 1.5),
     (5.0, 1.0), (6.3, 1.6), (7.7, 2.3), (6.4, 1.9), (6.5, 2.2), (5.7, 1.2), (6.9, 2.3), (5.7, 1.3), (6.1, 1.2),
     (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), (7.9, 2.0), (5.6, 1.1), (7.2, 1.8), (5.5, 1.3), (7.2, 1.6), (6.3, 2.5),
     (6.3, 1.8), (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), (5.5, 1.3), (5.5, 1.1), (5.9, 1.5), (6.0, 1.5),
     (5.9, 1.8)])
y_train = torch.FloatTensor(
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1,
     1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
     1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1])

model = ClassModel()  # здесь создавайте модель
# переведите модель в режим обучения

total = x_train.size(0)  # размер обучающей выборки
N = 1000  # число итераций алгоритма SGD
batch_size = 8  # размер мини-батча

optimizer = optim.Adam(params=model.parameters(), lr=0.01)  # оптимизатор Adam с шагом обучения lr=0.01
loss_func = torch.nn.BCEWithLogitsLoss()  # функция потерь: бинарная кросс-энтропия, класс nn.BCEWithLogitsLoss

for _ in range(N):
    idx = np.random.choice(total, batch_size, False)  # выбор индексов образов в размере batch_size
    # с помощью списочной индексации отберите из выборки x_train образы согласно индексам списка idx
    # пропустите через модель batch образов выборки и вычислите batch прогнозов predict
    y = model(x_train[idx]).flatten()
    loss = loss_func(
        y,
        y_train[idx],
    )  # вычислите значение функции потерь и сохраните результат в переменной loss

    # выполните один шаг градиентного спуска так, как это было сделано в предыдущем подвиге
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# переведите модель в режим эксплуатации
model.eval()
# прогоните через модель обучающую выборку и подсчитайте долю верных классификаций
# результат (долю верных классификаций) сохраните в переменной Q (в виде вещественного числа, а не тензора)
with torch.no_grad():
    predict = (model(x_train).flatten() > 0).float()

Q = (y_train == predict).float().mean()

```
