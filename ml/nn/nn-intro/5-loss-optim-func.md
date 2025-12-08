# Виды ф-ций оптимизации

задачи регрессии - nn.MSELoss, nn.L1Loss
задачи бинарной классификации - nn.BCELoss, nn.BCEWithLogitsLoss
задачи многоклассовой классификации - nn.CrossEntropyLoss, nn.NLLLoss

```python
import torch
import torch.optim as optim

w = torch.tensor([1, 2, 3])
m = 1

# функции оптимизаторы
optim.SGD(params=[w], lr=0.1, momentum=m, nesterov=True)
optim.RMSprop(params=[w], lr=0.1, momentum=m, )
optim.Adadelta(params=w, lr=0.1)
optim.Adam(params=w, lr=0.1)
```

# Вычисление точки минимума

```python
import torch
import torch.optim as optim


def func(x):
    return 0.2 * (x - 2) ** 2 - 0.3 * torch.cos(4 * x)


lr = 0.1  # шаг обучения
x0 = 0.0  # начальное значение точки минимума
N = 200  # число итераций градиентного алгоритма
x = torch.tensor([x0], requires_grad=True)

optimizer = optim.RMSprop(lr=lr, params=[x])

for _ in range(200):
    y = func(x)
    y.backward()

    optimizer.step()
    optimizer.zero_grad()

```

# Пример SGD

```python
import torch
import torch.optim as optim

from random import randint


def model(X, w):
    return X @ w


N = 2
w = torch.FloatTensor(N).uniform_(-1e-5, 1e-5)
w.requires_grad_(True)
x = torch.arange(0, 3, 0.1)

y_train = 0.5 * x + 0.2 * torch.sin(2 * x) - 3.0
x_train = torch.tensor([[_x ** _n for _n in range(N)] for _x in x])

total = len(x)
lr = torch.tensor([0.1, 0.01])
loss_func = torch.nn.L1Loss()  # ф-ция потерь
optimizer = optim.Adam(params=[w], lr=0.01)  # параметры ф-ции для оптимизации

for _ in range(1000):
    k = randint(0, total - 1)
    y = model(x_train[k], w)  # выбор batch
    loss = loss_func(y, y_train[k])  # использование оптимизатора

    loss.backward()  # вычисление градиента
    optimizer.step()  # w.data = w.data - lr * w.grad
    optimizer.zero_grad()  # w.grad.zero_() 

print(w)
predict = model(x_train, w)
```

```python
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam


def model(x, w1, w2, b1, b2):
    x = x @ w1.permute(1, 0) + b1
    x = torch.tanh(x)
    x = x @ w2.permute(1, 0) + b2
    return x


np.random.seed(1)  # установка "зерна" генератора датчика случайных чисел
torch.manual_seed(123)

W1 = torch.empty(2, 2).normal_(0, 1e-5)
bias1 = torch.rand(2, requires_grad=True)
W2 = torch.empty(1, 2).normal_(0, 1e-5)
bias2 = torch.rand(1, requires_grad=True)

W1.requires_grad_(True)
W2.requires_grad_(True)

# обучающая выборка
n_items = 20
C00 = torch.empty(n_items, 2).normal_(0, 1)
C11 = torch.empty(n_items, 2).normal_(0, 1) + torch.FloatTensor([5, 5])
C01 = torch.empty(n_items, 2).normal_(0, 1) + torch.FloatTensor([0, 5])
C10 = torch.empty(n_items, 2).normal_(0, 1) + torch.FloatTensor([5, 0])

x_train = torch.cat([C00, C11, C01, C10])
y_train = torch.cat([torch.ones(n_items * 2), torch.zeros(n_items * 2)])

lr = 0.01  # шаг обучения
N = 1000  # число итераций при обучении
total = y_train.size(0)  # размер обучающей выборки

# здесь продолжайте программу
loss_func = BCEWithLogitsLoss()
optimizer = Adam(params=[W1, W2, bias1, bias2], lr=lr)
for _ in range(N):
    k = np.random.randint(0, total)
    y = model(x_train[k], W1, W2, bias1, bias2)
    loss = loss_func(y[0], y_train[k])
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

predict = model(x_train, W1, W2, bias1, bias2).sum(dim=1) > 0

Q = (predict.float() == y_train).float().mean()
```