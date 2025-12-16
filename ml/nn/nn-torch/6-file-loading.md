Сохранение и загрузка файлов

```python
import torch
from collections import OrderedDict

# эти тензоры в программе не менять
layer1 = torch.rand(64, 32)
bias1 = torch.rand(32)
layer2 = torch.rand(32, 10)
bias2 = torch.rand(10)

# здесь продолжайте программу
data_w = OrderedDict({
    'layer1': layer1,
    'bias1': bias1,
    'layer2': layer2,
    'bias2': bias2
})
# сохранение файла
torch.save(data_w, 'data_w.tar')
# чтение файла
data_w2 = torch.load("data_w.tar")
```

Загрузка модели в память

```python
import torch
import torch.nn as nn


# здесь продолжайте программу
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(13, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        return self.layer3(x)


model = Model()
# преобразование для сохранения
st = model.state_dict()
# сохранение
torch.save(st, 'func_nn.tar')
```

Загрузка и составление прогноза

```python
import torch
import torch.nn as nn

# здесь объявляйте класс модели

x = torch.ones(48)  # тензор в программе не менять


# здесь продолжайте программу
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(48, 32, bias=False)
        self.layer2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 10)

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        return self.out(x)


model = Model()
# загрузка параметров модели из-вне
data_w2 = torch.load('toy_nn.tar', weights_only=True)
# загрузка параметров в модель
model.load_state_dict(data_w2)
predict = model(x)
```

Сохранение параметров модели (оптимизатора, потерь)

```python
import torch
import torch.nn as nn
import torch.optim as optim


# здесь продолжайте программу
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(25, 16, bias=False)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 5, bias=False)

    def forward(self, x):
        x = self.layer1(x).tahn()
        x = self.layer2(x).tahn()
        return self.layer3(x)


model = Model()

opt = optim.Adam(params=model.parameters(), lr=0.02)
loss_func = nn.CrossEntropyLoss()

data_state_dict = {
    'loss': loss_func.state_dict(),
    'opt': opt.state_dict(),
    'model': model.state_dict(),
}

torch.save(data_state_dict, 'my_data_state.tar')

```

Загрузка параметров модели (оптимизатора, потерь), данные в виде data_state_dict:

```python

data_state_dict = {
    'opt': 'состояние оптимизатора',
    'model': 'состояние модели (веса)',
}
```

```python
import torch
import torch.nn as nn
import torch.optim as optim


# здесь продолжайте программу
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 8)
        self.layer2 = nn.Linear(8, 4)
        self.out = nn.Linear(4, 6)

    def forward(self, x):
        x = self.layer1(x).sigmoid()
        x = self.layer2(x).sigmoid()
        return self.out(x)


model = Model()

opt = optim.RMSprop(params=model.parameters(), lr=0.05)

data_state_dict = torch.load('nn_data_state.tar')
model.load_state_dict(data_state_dict['model'])
opt.load_state_dict(data_state_dict['opt'])
```