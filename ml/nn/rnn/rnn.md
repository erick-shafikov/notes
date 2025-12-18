# RNN

[RNN](../../libs/pytorch/models/rnn.md)

# MTO, MTM

## mto

модель model структуры типа Many-to-One,

```python
import torch
import torch.nn as nn

batch_size = 4
seq_length = 8
in_features = 10
x = torch.rand(batch_size, seq_length, in_features)


class GetOutput(nn.Module):
    def forward(self, x):
        _, h = x # вернем результат
        return h


model = nn.Sequential(
    nn.RNN(in_features, 15, batch_first=True),
    GetOutput(),
    nn.ReLU(inplace=True),
    nn.Linear(15, 5, bias=True)
)

model.eval()
res = model(x)
```

модель model структуры типа Many-to-One если num_layers=2

```python
import torch
import torch.nn as nn



class OutputToLinear(nn.Module):
    def forward(self, x):
        return x[1][1] # берем последний с последнего



batch_size = 18
seq_length = 21
in_features = 5
x = torch.rand(batch_size, seq_length, in_features)

# здесь продолжайте программу
model = nn.Sequential(
    nn.RNN(in_features, 25, batch_first=True, num_layers=2),
    OutputToLinear(),
    nn.ReLU(inplace=True),
    nn.Linear(25, 5, bias=True)
)

model.eval()
predict = model(x)

```

## mtm

модель model структуры типа Many-to-Many

```python
import torch
import torch.nn as nn


# здесь объявляйте класс OutputModule
class OutputModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(25, 10, True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.layer(self.act(x[0])) # вернем все скрытые



batch_size = 7
seq_length = 5
in_features = 15
x = torch.rand(batch_size, seq_length, in_features)

# здесь продолжайте программу
model = nn.Sequential(
    nn.RNN(in_features, 25, batch_first=True),
    OutputModule()
)

model.eval()
out = model(x) # [7, 5, 10]
```

# Примеры работы с RNN

## Вычисление по формуле (одномерной)

x[t] = tanh(r _ x[t-1] + sigma _ sigma_noise)

```python
import torch
import torch.nn as nn

sigma = 0.1  # стандартное отклонение отсчетов последовательности
r = 0.9  # коэффициент регрессии
sigma_noise = sigma * (1 - r * r) ** 0.5  # стандартное отклонение случайных величин

total = 100  # длина генерируемой последовательности
noise = torch.randn((total, 1, 1))  # случайные величины, подаваемые на вход модели
x0 = torch.randn((1, 1, 1)) * sigma  # начальное значение вектора скрытого состояния

model = nn.RNN(1, 1, bias=False)  # создание RNN вход - 1 число из вектора, 1 - размер внутреннего сигнала
# Изменения весов
model.weight_hh_l0.data = torch.tensor([[r]])
model.weight_ih_l0.data = torch.tensor([[sigma_noise]])
model.eval()

x, _ = model(noise, x0)
```

## Вычисление по формуле (двумерной)

[
x[t] = tanh(r _ x[t-1] + sigma _ sigma*noise),
y[t] = tanh(r * y[t-1] + sigma \_ sigma_noise)
]

```python
import torch
import torch.nn as nn

sigma_x, sigma_y = 0.1, 0.15  # стандартные отклонения отсчетов последовательности
rx, ry = 0.9, 0.99  # # коэффициенты регрессии
sigma_noise_x = sigma_x * (1 - rx * rx) ** 0.5  # стандартное отклонение случайных величин
sigma_noise_y = sigma_y * (1 - ry * ry) ** 0.5  # стандартное отклонение случайных величин

total = 100  # длина генерируемой последовательности
noise = torch.randn((total, 1, 2))  # случайные величины, подаваемые на вход модели
h0 = torch.randn((1, 1, 2)) * torch.tensor(
    [sigma_noise_x, sigma_noise_y])  # начальное значение вектора скрытого состояния

# здесь продолжайте программу
model = nn.RNN(2, 2, bias=False)
model.weight_hh_l0.data = torch.tensor([[rx, 0], [0, ry]])
model.weight_ih_l0.data = torch.tensor([[sigma_noise_x, 0], [0, sigma_noise_y]])

model.eval()
x, _ = model(noise, h0)
```

# обучение

## Обучение прогноза математической функции

```python
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


# здесь объявляйте класс модели
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(input_size=1, hidden_size=5, batch_first=True)
        self.out = nn.Linear(in_features=5, out_features=1, bias=True)

    def forward(self, x):
        _, ht = self.rnn(x)
        return self.out(ht)


x = torch.linspace(-20, 20, 2000)
y = torch.cos(x) + 0.5 * torch.sin(5 * x) + 0.1 * torch.randn_like(x)

total = len(x)  # общее количество отсчетов
train_size = 1000  # размер обучающей выборки
seq_length = 10  # число предыдущих отсчетов, по которым строится прогноз следующего значения

y.unsqueeze_(1)
train_data_y = torch.cat([y[i:i + seq_length] for i in range(train_size - seq_length)], dim=1)
train_targets = torch.tensor([y[i + seq_length].item() for i in range(train_size - seq_length)])

test_data_y = torch.cat([y[i:i + seq_length] for i in range(train_size - seq_length, total - seq_length)], dim=1)
test_targets = torch.tensor([y[i + seq_length].item() for i in range(train_size - seq_length, total - seq_length)])

d_train = data.TensorDataset(train_data_y.permute(1, 0), train_targets)
d_test = data.TensorDataset(test_data_y.permute(1, 0), test_targets)

train_data = data.DataLoader(d_train, batch_size=8, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=len(d_test), shuffle=False)

model = Model()  # создание объекта модели

optimizer = optim.RMSprop(params=model.parameters(), lr=0.001)  # оптимизатор RMSprop с шагом обучения 0.001
loss_func = nn.MSELoss()  # функция потерь - средний квадрат ошибок

epochs = 5  # число эпох
# переведите модель в режим обучения
model.train()
lass_lost = 0
for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train.unsqueeze(2))  # вычислите прогноз модели для x_train
        loss = loss_func(predict.squeeze(), y_train)  # вычислите потери для predict и y_train

        # выполните один шаг обучения (градиентного спуска)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lass_lost = loss

# переведите модель в режим эксплуатации
model.eval()
d, t = next(iter(test_data))

# с использованием менеджера torch.no_grad вычислите прогнозы для выборки d
with torch.no_grad():
    # результат сохраните в тензоре predict
    predict = model(d.unsqueeze(2))
    Q = loss_func(
        predict.squeeze(),
        t,
    ).item()  # вычислите потери с помощью loss_func для predict и t; значение Q сохраните в виде вещественного числа

```

## Обучение на текстовом файле

```python
import re

from tqdm import tqdm
import torch

import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class CharsDataset(data.Dataset):
    def __init__(self, path, prev_chars=3):
        self.prev_chars = prev_chars

        with open(path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            self.text = self.text.replace('\ufeff', '')  # убираем первый невидимый символ
            self.text = re.sub(r'[^А-яA-z0-9.,?;: ]', '',
                               self.text)  # заменяем все неразрешенные символы на пустые символы

        self.text = self.text.lower()
        self.alphabet = set(self.text)
        self.int_to_alpha = dict(enumerate(sorted(self.alphabet)))
        self.alpha_to_int = {b: a for a, b in self.int_to_alpha.items()}
        # self.alphabet = {'а': 0, 'б': 1, 'в': 2, 'г': 3, 'д': 4, 'е': 5, 'ё': 6, 'ж': 7, 'з': 8, 'и': 9,
        #                  'й': 10, 'к': 11, 'л': 12, 'м': 13, 'н': 14, 'о': 15, 'п': 16, 'р': 17, 'с': 18,
        #                  'т': 19, 'у': 20, 'ф': 21, 'х': 22, 'ц': 23, 'ч': 24, 'ш': 25, 'щ': 26, 'ъ': 27,
        #                  'ы': 28, 'ь': 29, 'э': 30, 'ю': 31, 'я': 32, ' ': 33, '.': 34, '!': 35, '?': 36}
        self.num_characters = len(self.alphabet)
        self.onehots = torch.eye(self.num_characters)

    def __getitem__(self, item):
        _data = torch.vstack(
            [self.onehots[self.alpha_to_int[self.text[x]]] for x in range(item, item + self.prev_chars)])
        ch = self.text[item + self.prev_chars]
        t = self.alpha_to_int[ch]
        return _data, t

    def __len__(self):
        return len(self.text) - 1 - self.prev_chars


class TextRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 64
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.out(h)
        return y


d_train = CharsDataset("train_data_true", prev_chars=10)
train_data = data.DataLoader(d_train, batch_size=8, shuffle=False)

model = TextRNN(d_train.num_characters, d_train.num_characters)

optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

epochs = 100
model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train).squeeze(0)
        loss = loss_func(predict, y_train.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

st = model.state_dict()
torch.save(st, 'model_rnn_1.tar')

# st = torch.load('model_rnn_1.tar', weights_only=False)
# model.load_state_dict(st)

model.eval()
predict = "Мой дядя самых".lower()
total = 40

for _ in range(total):
    _data = torch.vstack([d_train.onehots[d_train.alpha_to_int[predict[-x]]] for x in range(d_train.prev_chars, 0, -1)])
    p = model(_data.unsqueeze(0)).squeeze(0)
    indx = torch.argmax(p, dim=1)
    predict += d_train.int_to_alpha[indx.item()]

print(predict)

```

## С прогнозом на 20 символов

```python
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

_global_var_text = [
    "Как я отмечал во введении, простейшая НС – персептрон, представляет собой ...",
    "Это классический пример полносвязной сети ...",
    "Каждая связь между нейронами имеет определенный ...",
]


class CharsDataset(data.Dataset):
    def __init__(self, prev_chars):
        self.prev_chars = prev_chars
        self.int_to_alpha = sorted(set(''.join(_global_var_text).lower()))
        self.num_characters = len(self.int_to_alpha)
        self.alpha_to_int = dict(zip(self.int_to_alpha, range(self.num_characters)))
        self.onehots = torch.eye(self.num_characters)

        data = []
        targets = []

        for row in _global_var_text:
            line = torch.tensor([self.alpha_to_int[ch] for ch in row.lower()])
            for i in range(len(line) - self.prev_chars):
                data.append(line[i:i + self.prev_chars].view(1, -1))
            targets.append(line[self.prev_chars:])

        self.data = torch.cat(data)
        self.targets = torch.cat(targets)

    def __getitem__(self, item):
        return self.onehots[self.data[item]], self.targets[item]

    def __len__(self):
        return len(self.data)


class TextRNN(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.rnn = nn.RNN(features, 32, batch_first=True)
        self.out = nn.Linear(32, features)

    def forward(self, x):
        return self.out(self.rnn(x)[1])


# сюда копируйте объекты d_train и train_data
d_train = CharsDataset(prev_chars=10)
train_data = data.DataLoader(d_train, batch_size=8, shuffle=True)

model = TextRNN(d_train.num_characters)  # создайте объект модели

optimizer = optim.Adam(model.parameters(), lr=0.01)  # оптимизатор Adam с шагом обучения 0.01
loss_func = nn.CrossEntropyLoss()  # функция потерь - CrossEntropyLoss

epochs = 1  # число эпох (это конечно, очень мало, в реальности нужно от 100 и более)
# переведите модель в режим обучения
model.train()

for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train).squeeze(0)  # вычислите прогноз модели для x_train
        loss = loss_func(predict, y_train)  # вычислите потери для predict и y_train

        # выполните один шаг обучения (градиентного спуска)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# переведите модель в режим эксплуатации
model.eval()
predict = "нейронная сеть ".lower()  # начальная фраза
total = 20  # число прогнозируемых символов (дополнительно к начальной фразе)

# выполните прогноз следующих total символов
for _ in range(total):
    data = torch.stack([d_train.onehots[d_train.alpha_to_int[x]] for x in predict[-d_train.prev_chars:]])
    p = model(data.unsqueeze(0)).squeeze()
    indx = torch.argmax(p)
    predict += d_train.int_to_alpha[indx.item()]

# выведите полученную строку на экран
print(predict)

```
