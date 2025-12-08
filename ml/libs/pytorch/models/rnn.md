# RNN нативная реализация

Схема:

- Linear(16, 10, True)
- рекуррентный
  слой:
  -- h[t - 1] +
  -- tanh
  -- h[t]
- Linear(10, 5, True)
- Sigmoid()

```python
import torch
import torch.nn as nn

batch_size = 8  # размер батча
seq_length = 6  # длина последовательности
in_features = 16  # размер каждого элемента последовательности
x = torch.rand(batch_size, seq_length, in_features)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(16, 10)
        self.layer2 = nn.Linear(10, 5)

    def forward(self, x):
        h = 0
        for i in range(x.size(1)):
            y = self.layer1(x[:, i, :])
            y += h
            y = torch.tanh(y)
            h = y
        y = self.layer2(y)
        y = torch.sigmoid(y)
        return y


model = Model()
model.eval()

out = model(x)
```

# nn.RNN

```python
import torch
from torch.nn import RNN

RNN(
    input_size=33,  # размер данных (отдельные элементы) для входного тензора 33 в случе кириллицы
    hidden_size=64,  # размер вектора скрытого состояния
    num_layers=1,  # количество слоев 
    nonlinearity='tanh',  # ф-ция активации
    bias=True,
    batch_first=True,  # режим батча по умолчанию False
    biderectional=False  # True если нужна двунаправленная rnn
)

# формат входного тензора
# если batch_first=False в RNN нужно менять местами seq_length и batch_size
# batch_size = 8 количество наборов токенов, размер батча, количество слов
# seq_length = 3 последовательность из токенов (количество букв при обработке)
# d_size = 33 размер матрицы для одного токена (буквы)
x = torch.randn(8, 3, 33)
# первоначальный h[t-1] слой
# всегда один
# для каждого батча
# внутреннее представление одного токена
h0 = torch.randn(1, 8, 64)

rnn = RNN(input_size=33, hidden_size=64, batch_first=True)  # input_size === d_size

y, h = rnn(x, h0)  # y - вектор результатов с каждой итерации, h - итоговый

print(y.size())  # torch.Size([8, 3, 64]) 
print(h.size())  # torch.Size([1, 8, 64]) если слоев > 1 то на выходе будет [n, 8, 64]
``` 

## bidirectional

```python
import torch
from torch import nn

rnn = nn.RNN(input_size=300, hidden_size=16, bidirectional=True)

# y содержит concat-слои h - результат
# y : [8,3,32]
# h : [2,8,16] - первый вектор получен при прямом проходе, второй при обратном
y, h = rnn(torch.randn(8, 3, 300))
```

Получения результатов контекста прямого прохода и обратного

```python
import torch
import torch.nn as nn


# здесь объявляйте класс модели
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(32, 12, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(24, 5, bias=True)

    def forward(self, x):
        _, h = self.rnn(x)
        # h[0] - прямой проход, h[1] - обратный проход
        hh = torch.cat((h[0], h[1]), dim=1)
        return self.linear(hh)


batch_size = 8
seq_length = 12
d_size = 32
x = torch.rand(batch_size, seq_length, d_size)

model = Model()  # создание объекта модели

# здесь продолжайте программу
model.eval()
predict = model(x)
```

Модель many to many

```python
import torch
import torch.nn as nn


# здесь объявляйте класс OutputModule
class OutputModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(8, 2, bias=False)

    def forward(self, x):
        h, _ = x
        # берем с каждого скрытого слоя результат прогона через линейную модель объединяем в один тензор
        return torch.cat(
            [
                self.layer(h[:, i, :].unsqueeze(1)) for i in range(h.size(1))
            ],
            dim=1)


# тензор x в программе не менять
batch_size = 4
seq_length = 64
in_features = 5
x = torch.rand(batch_size, seq_length, in_features)

model = nn.Sequential(
    nn.RNN(5, 4, batch_first=True, bidirectional=True),
    OutputModule()
)  # создание объекта модели

# здесь продолжайте программу
model.eval()

out = model(x)
```

Модель one-to-many

```python
import torch
import torch.nn as nn


# здесь объявляйте класс модели
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(5, 7, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(14, 2)

    def forward(self, x):
        h, y = self.rnn(x)
        return self.linear(h)


# тензор x в программе не менять
batch_size = 4
seq_length = 12
d_size = 5
x = torch.rand(batch_size, d_size)

X = torch.zeros(batch_size, seq_length, d_size)
X[:, 0, :] = x

model = Model()  # создание объекта модели

# здесь продолжайте программу
model.eval()

predict = model(X)
```

комбинация рекуррентных слоев типа Many-to-One и One-to-Many

```python
import torch
import torch.nn as nn


# здесь объявляйте класс модели
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.brnn = nn.RNN(5, 9, bidirectional=True, batch_first=True)
        self.rnn = nn.RNN(18, 32, batch_first=True)
        self.linear = nn.Linear(32, 3, bias=True)

    def forward(self, x):
        _, y = self.brnn(x)

        output_1 = torch.cat([y[0, :, :].squeeze(1), y[1, :, :].squeeze(1)], dim=1)
        input_2 = torch.zeros(2, 25, 18)
        input_2[:, 0, :] = output_1

        h, _ = self.rnn(input_2)

        return self.linear(h)


# тензор x в программе не менять
batch_size = 2
seq_length = 12
in_features = 5
x = torch.rand(batch_size, seq_length, in_features)

model = Model()  # создание объекта модели

# здесь продолжайте программу
model.eval()

results = model(x)

```