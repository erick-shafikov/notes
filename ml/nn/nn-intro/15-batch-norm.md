# batch normalization

применение BatchNorm1d

```python
import torch
import torch.nn as nn


# здесь объявляйте класс модели (важно именно здесь)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 64, bias=False)
        self.out = nn.Linear(64, 1)
        # ф-ция batch normalization
        self.bn = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer1(x).relu()
        # использование после функции активации
        x = self.bn(x)
        return self.out(x)


model = Model()  # здесь создавайте объект модели
model.eval()

batch_size = 16
x = torch.rand(batch_size, 10)  # этот тензор в программе не менять

# здесь продолжайте программу
predict = model(x)

```

```python
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class RavelTransform(nn.Module):
    def forward(self, item):
        return item.ravel()


class DigitNN(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        # bias=False отключаем bias
        self.layer1 = nn.Linear(input_dim, num_hidden, bias=False)
        self.layer2 = nn.Linear(num_hidden, output_dim)
        # добавление возможности для BatchNorm
        self.bm_1 = nn.BatchNorm1d(num_hidden)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        # применение
        x = self.bm_1(x)
        x = self.layer2(x)
        return x


model = DigitNN(28 * 28, 128, 10)

transforms = tfs.Compose([tfs.ToImage(), tfs.Grayscale(),
                          tfs.ToDtype(torch.float32, scale=True),
                          RavelTransform(),
                          ])

dataset_mnist = torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=True, transform=transforms)
d_train, d_val = data.random_split(dataset_mnist, [0.7, 0.3])
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False)

optimizer = optim.Adam(params=model.parameters(), lr=0.01)  # , weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()
epochs = 20

loss_lst_val = []
loss_lst = []

for _e in range(epochs):
    model.train()
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=False)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

    # валидация модели
    model.eval()
    Q_val = 0
    count_val = 0

    for x_val, y_val in train_data_val:
        with torch.no_grad():
            p = model(x_val)
            loss = loss_function(p, y_val)
            Q_val += loss.item()
            count_val += 1

    Q_val /= count_val

    loss_lst.append(loss_mean)
    loss_lst_val.append(Q_val)

    print(f" | loss_mean={loss_mean:.3f}, Q_val={Q_val:.3f}")

d_test = torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=False, transform=transforms)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)

Q = 0

model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        Q += torch.sum(p == y_test).item()

Q /= len(d_test)
print(Q)

# вывод графиков
plt.plot(loss_lst)
plt.plot(loss_lst_val)
plt.grid()
plt.show()
```

Пример с большим количеством

```python
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer

_data = load_breast_cancer()

_global_var_data_x = torch.tensor(_data.data, dtype=torch.float32)
_global_var_target = torch.tensor(_data.target, dtype=torch.int64)


# здесь продолжайте программу
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(30, 32, bias=False)
        self.layer2 = nn.Linear(32, 20, bias=False)
        self.layer3 = nn.Linear(20, 1)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.batch_norm1(x)
        x = self.layer2(x).relu()
        x = self.batch_norm2(x)

        return self.layer3(x)


model = Model()
model.train()

ds = data.TensorDataset(_global_var_data_x, _global_var_target.float())
train_d, test_d = data.random_split(ds, [0.7, 0.3])

train_data = data.DataLoader(train_d, batch_size=16, shuffle=True)
test_data = data.DataLoader(test_d, batch_size=len(test_d))

optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_func = nn.BCEWithLogitsLoss()
epoches = 5

for _ in range(epoches):
    for train_x, train_y in train_data:
        predict = model(train_x).squeeze(1)
        loss = loss_func(predict, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

with torch.no_grad():
    for x_test, y_test in test_data:
        predict = (model(x_test).squeeze_() > 0).float()
        Q = (predict == y_test).float().mean()
```