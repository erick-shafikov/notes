```python
import os
import numpy as np
import re

from navec import Navec
from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim


class WordsDataset(data.Dataset):
    # path до весов предобученной модели
    # navec_emb
    def __init__(self, path, navec_emb, prev_words=3):
        self.prev_words = prev_words
        self.navec_emb = navec_emb

        with open(path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            self.text = self.text.replace('\ufeff', '')  # убираем первый невидимый символ
            self.text = self.text.replace('\n', ' ')
            self.text = re.sub(r'[^А-яA-z- ]', '', self.text)  # удаляем все неразрешенные символы

        self.words = self.text.lower().split()
        self.words = [word for word in self.words if word in self.navec_emb]  # оставляем слова, которые есть в словаре
        vocab = set(self.words)

        self.int_to_word = dict(enumerate(vocab))
        self.word_to_int = {b: a for a, b in self.int_to_word.items()}
        self.vocab_size = len(vocab)

    def __getitem__(self, item):
        _data = torch.vstack([torch.tensor(self.navec_emb[self.words[x]]) for x in range(item, item + self.prev_words)])
        word = self.words[item + self.prev_words]
        t = self.word_to_int[word]
        return _data, t

    def __len__(self):
        return len(self.words) - 1 - self.prev_words


class WordsRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 256
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.out(h)
        return y


path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

d_train = WordsDataset("text_2", navec, prev_words=3)
train_data = data.DataLoader(d_train, batch_size=8, shuffle=False)

model = WordsRNN(300, d_train.vocab_size)

optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)
loss_func = nn.CrossEntropyLoss()

epochs = 20
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
torch.save(st, 'model_rnn_words.tar')

model.eval()
predict = "подумал встал и снова лег".lower().split()
total = 10

for _ in range(total):
    _data = torch.vstack([torch.tensor(d_train.navec_emb[predict[-x]]) for x in range(d_train.prev_words, 0, -1)])
    p = model(_data.unsqueeze(0)).squeeze(0)
    indx = torch.argmax(p, dim=1)
    predict.append(d_train.int_to_word[indx.item()])

print(" ".join(predict))
```

Еще один прогноз с navec

```python
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from navec import Navec

#######################################################################
path = 'datasets/navec_hudlit_v1_12B_500K_300d_100q.tar'
global_navec = Navec.load(path)

# is_word = 'каждый' in global_navec  # проверка наличия слова в словаре
# v = global_navec['каждый']  # embedding.md слова 'каждый'
# indx = global_navec.vocab['каждый']  # индекс слова 'каждый' в словаре

#######################################################################

_global_var_text = [
    "Как я отмечал во введении простейшая представляет собой",
    "Это классический пример сети",
    "Каждая связь между нейронами имеет определенный",

]


# сюда копируйте класс WordsDataset, созданный на предыдущем занятии
class WordsDataset(data.Dataset):
    def __init__(self, prev_words):
        self.prev_words = prev_words

        self.data = [
            {
                'prev_words': words[i:i + prev_words],
                'predict': words[i + prev_words]
            }
            for row in _global_var_text
            for words in [row.lower().split()]
            for i in range(len(words) - prev_words)
        ]

    def __getitem__(self, item):
        data_item = self.data[item]

        return (
            torch.stack([torch.tensor(global_navec[word]) for word in data_item['prev_words']]),
            torch.tensor(global_navec.vocab[data_item['predict']])
        )

    def __len__(self):
        return len(self.data)


# здесь объявляйте класс модели
class Model(nn.Module):
    def __init__(self, input_size=300, out_features=100):
        super().__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=16, batch_first=True)
        self.out = nn.Linear(in_features=16, out_features=out_features)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.out(h)


# сюда копируйте объекты d_train и train_data, созданные на предыдущем занятии
prev_words = 4
d_train = WordsDataset(prev_words)
train_data = data.DataLoader(d_train, batch_size=8, shuffle=True)

# создайте объект модели для прогноза слов
model = Model(
    # input_size=100,
    input_size=300,
    # out_features=len(global_navec.vocab),
    out_features=len(global_navec.vocab.words)
)

# создайте оптимизатор Adam с шагом обучения 0.01 и параметром weight_decay=0.0001
optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.0001)
loss_func = nn.CrossEntropyLoss()  # создайте функцию потерь кросс-энтропию для задачи многоклассовой классификации

epochs = 100  # число эпох обучения (в реальности нужно от 100 и более)
# переведите модель в режим обучения
model.train()

for _e in range(epochs):
    # с помощью цикла for переберите батчи из объекта train_data
    for x, y in train_data:
        predict = model(x).squeeze(0)  # вычислите прогноз модели для x_train
        loss = loss_func(predict, y.long())  # вычислите потери для predict и y_train

        # выполните один шаг обучения (градиентного спуска)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# переведите модель в режим эксплуатации
model.eval()
predict = "Такими были первые нейронные сети предложенные".lower().split()
total = 10  # число прогнозируемых слов (дополнительно к начальной фразе)

# выполните прогноз следующих total слов
for _ in range(10):
    _data = torch.stack([torch.tensor(global_navec[predict[-x]]) for x in range(d_train.prev_words, 0, -1)])
    p = model(_data.unsqueeze(0)).squeeze(0)
    indx = torch.argmax(p, dim=1).item()

    predict.append(global_navec.vocab.words[indx])

# выведите полученную строку на экран
print(" ".join(predict))

```