Применения LTS

```python

import re
from navec import Navec
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class PhraseDataset(data.Dataset):
    def __init__(self, path_true, path_false, navec_emb, batch_size=8):
        self.navec_emb = navec_emb
        self.batch_size = batch_size

        with open(path_true, 'r', encoding='utf-8') as f:
            phrase_true = f.readlines()
            self._clear_phrase(phrase_true)

        with open(path_false, 'r', encoding='utf-8') as f:
            phrase_false = f.readlines()
            self._clear_phrase(phrase_false)

        self.phrase_lst = [(_x, 0) for _x in phrase_true] + [(_x, 1) for _x in phrase_false]
        self.phrase_lst.sort(key=lambda _x: len(_x[0]))
        self.dataset_len = len(self.phrase_lst)

    def _clear_phrase(self, p_lst):
        for _i, _p in enumerate(p_lst):
            _p = _p.lower().replace('\ufeff', '').strip()
            _p = re.sub(r'[^А-яA-z- ]', '', _p)
            _words = _p.split()
            _words = [w for w in _words if w in self.navec_emb]
            p_lst[_i] = _words

    def __getitem__(self, item):
        item *= self.batch_size
        item_last = item + self.batch_size
        if item_last > self.dataset_len:
            item_last = self.dataset_len

        _data = []
        _target = []
        max_length = len(self.phrase_lst[item_last - 1][0])

        for i in range(item, item_last):
            words_emb = []
            phrase = self.phrase_lst[i]
            length = len(phrase[0])

            for k in range(max_length):
                t = torch.tensor(self.navec_emb[phrase[0][k]], dtype=torch.float32) if k < length else torch.zeros(300)
                words_emb.append(t)

            _data.append(torch.vstack(words_emb))
            _target.append(torch.tensor(phrase[1], dtype=torch.float32))

        _data_batch = torch.stack(_data)
        _target = torch.vstack(_target)
        return _data_batch, _target

    def __len__(self):
        last = 0 if self.dataset_len % self.batch_size == 0 else 1
        return self.dataset_len // self.batch_size + last


class WordsRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 16
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.LSTM(in_features, self.hidden_size, batch_first=True, bidirectional=True)
        self.out = nn.Linear(self.hidden_size * 4, out_features)

    def forward(self, x):
        x, (h, c) = self.rnn(x)
        hh = torch.cat((h[-2, :, :], h[-1, :, :], c[-2, :, :], c[-1, :, :]), dim=1)
        y = self.out(hh)
        return y


path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

d_train = PhraseDataset("train_data_true", "train_data_false", navec)
train_data = data.DataLoader(d_train, batch_size=1, shuffle=True)

model = WordsRNN(300, 1)

optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
loss_func = nn.BCEWithLogitsLoss()

epochs = 20
model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train.squeeze(0)).squeeze(0)
        loss = loss_func(predict, y_train.squeeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

st = model.state_dict()
torch.save(st, 'model_rnn_bidir.tar')

model.eval()

phrase = "Сегодня пасмурная погода"
phrase_lst = phrase.lower().split()
phrase_lst = [torch.tensor(navec[w]) for w in phrase_lst if w in navec]
_data_batch = torch.stack(phrase_lst)
predict = model(_data_batch.unsqueeze(0)).squeeze(0)
p = torch.nn.functional.sigmoid(predict).item()
print(p)
print(phrase, ":", "положительное" if p < 0.5 else "отрицательное")
```

Прогноз значений функций

```python
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


# здесь объявляйте класс LSTMToLinear
class LSTMToLinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y, (h, c) = x

        return h


model = nn.Sequential(
    nn.LSTM(1, 10, batch_first=True),
    LSTMToLinear(),
    nn.Linear(10, 1)
)  # здесь описывайте модель с помощью класса Sequential

x = torch.linspace(-10, 10, 2000)
y = torch.cos(x) + 0.5 * torch.sin(5 * x) + 0.1 * torch.randn_like(x) + 0.2 * x

total = len(x)  # общее количество отсчетов
train_size = 1000  # размер обучающей выборки
seq_length = 20  # число предыдущих отсчетов, по которым строится прогноз следующего значения

y.unsqueeze_(1)
train_data_y = torch.cat([y[i:i + seq_length] for i in range(train_size - seq_length)], dim=1)
train_targets = torch.tensor([y[i + seq_length].item() for i in range(train_size - seq_length)])

test_data_y = torch.cat([y[i:i + seq_length] for i in range(train_size - seq_length, total - seq_length)], dim=1)
test_targets = torch.tensor([y[i + seq_length].item() for i in range(train_size - seq_length, total - seq_length)])

d_train = data.TensorDataset(train_data_y.permute(1, 0), train_targets)
d_test = data.TensorDataset(test_data_y.permute(1, 0), test_targets)

train_data = data.DataLoader(d_train, batch_size=8, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=len(d_test), shuffle=False)

optimizer = optim.RMSprop(model.parameters(), lr=0.01)  # оптимизатор RMSprop с шагом обучения 0.01
loss_func = nn.MSELoss()  # функция потерь - средний квадрат ошибок

epochs = 5  # число эпох
# переведите модель в режим обучения
model.train()

# TODO epochs
for _e in range(1):
    for x_train, y_train in train_data:
        predict = model(x_train.unsqueeze(-1))  # вычислите прогноз модели для x_train
        loss = loss_func(predict.squeeze(-1), y_train.unsqueeze(0))  # вычислите потери для predict и y_train

        # выполните один шаг обучения (градиентного спуска)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# переведите модель в режим эксплуатации
model.eval()
d, t = next(iter(test_data))

# с использованием менеджера torch.no_grad вычислите прогнозы для выборки d
with torch.no_grad():
    predict = model(d.unsqueeze(-1))  # результат сохраните в тензоре predict
    Q = loss_func(predict.squeeze(-1), t.unsqueeze(
        0)).item()  # вычислите потери с помощью loss_func для predict и t; значение Q сохраните в виде вещественного числа

```