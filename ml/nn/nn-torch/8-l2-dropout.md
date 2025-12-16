# Dropout

- nn.Dropout() - для полносвязных слоев nn.Linear
- nn.Dropout1d() - Conv1d
- nn.Dropout2d() - Conv2d

Пример добавления

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.layer1 = nn.Linear(13, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 3)
        # dropout
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.layer1(x).relu()
        # пропуск через dropout
        x = self.dropout(x)
        x = self.layer2(x).relu()
        return self.layer3(x)


model = Model()  # здесь создавайте объект модели

x = torch.rand(13)

# здесь продолжайте программу для вычисления predict
predict = model(x)
```

# l2-регулизатор

```python
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import ImageFolder


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


model = DigitNN(28 * 28, 128, 10)

transforms = tfs.Compose(
    [
        tfs.ToImage(),
        tfs.Grayscale(),
        tfs.ToDtype(torch.float32, scale=True),
        RavelTransform(),
    ]
)

dataset_mnist = torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=True, transform=transforms)

d_train, d_val = data.random_split(dataset_mnist, [0.7, 0.3])
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False)

# weight_decay - ламбда коэффициент l2 регулеризатора
optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()
epochs = 20

loss_lst_val = []  # список значений потерь при валидации
loss_lst = []  # список значений потерь при обучении

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

Пример с обучением модели

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
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)
        # реализация Dropout (1d так как вектор у нас выпрямляется)
        self.dropout_1 = nn.Dropout1d(0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        # применение dropout_1
        x = self.dropout_1(x)
        x = self.layer2(x)
        return x


model = DigitNN(28 * 28, 128, 10)

transforms = tfs.Compose(
    [
        tfs.ToImage(),
        tfs.Grayscale(),
        tfs.ToDtype(torch.float32, scale=True),
        RavelTransform(),
    ]
)

dataset_mnist = torchvision.datasets.MNIST(r'C:\datasets\mnist', download=True, train=True, transform=transforms)
d_train, d_val = data.random_split(dataset_mnist, [0.7, 0.3])
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
train_data_val = data.DataLoader(d_val, batch_size=32, shuffle=False)

optimizer = optim.Adam(params=model.parameters(), lr=0.01)  # , weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()
epochs = 20

loss_lst_val = []  # список значений потерь при валидации
loss_lst = []  # список значений потерь при обучении

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

# тестирование обученной НС
model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        Q += torch.sum(p == y_test).item()

Q /= len(d_test)

```

# l2 + dropout

```python
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.datasets import load_digits

digits = load_digits()

# Признаки и целевые значения
_global_var_data_x = torch.tensor(digits.data, dtype=torch.float32)
_global_var_target = torch.tensor(digits.target, dtype=torch.int64)


# здесь объявляйте класс модели (обязательно до тензора x)
class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.dropout(x)
        x = self.layer2(x).relu()
        x = self.dropout(x)
        return self.layer3(x)


dataset = data.TensorDataset(_global_var_data_x, _global_var_target)
d_train, d_test = data.random_split(dataset, [0.7, 0.3])

train_data = data.DataLoader(d_train, batch_size=16, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=len(d_test))

epochs = 2
model = Model()

optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.1)
loss_func = nn.CrossEntropyLoss()

model.train()
for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train)
        loss = loss_func(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

x_test, y_test = next(iter(test_data))

with torch.no_grad():
    predict = model(x_test)
    predict = predict.argmax(dim=1)

Q = (predict == y_test).float().mean().item()

```
