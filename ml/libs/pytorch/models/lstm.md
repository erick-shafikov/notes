```python
# bidirectional=False | bidirectional=True
import torch
import torch.nn as nn

rnn = nn.LSTM(10, 16, batch_first=True, bidirectional=True)

x = torch.rand(1, 5, 10)

# y - результат
# h - скрытый слой
# c - контекст
y, (h, c) = rnn(x)

print(y.size())  # torch.Size([1, 5, 16]) | torch.Size([1, 5, 32])
print(h.size())  # torch.Size([1, 1, 16]) | torch.Size([2, 1, 16])
print(c.size())  # torch.Size([1, 1, 16]) | torch.Size([2, 1, 16])

```

# mto

```python
import torch
import torch.nn as nn


# здесь объявляйте класс LSTMToLinear
class LSTMToLinear(nn.Module):
    def forward(self, x):
        y, (h, c) = x

        return h


# тензор x в программе не менять
batch_size = 18
seq_length = 21
in_features = 5
x = torch.rand(batch_size, seq_length, in_features)

# здесь продолжайте программу
model = nn.Sequential(
    nn.LSTM(in_features, 25, batch_first=True),
    LSTMToLinear(),
    nn.Linear(25, 5)
)

model.eval()

res = model(x)

```

#mtm

```python
import torch
import torch.nn as nn

# здесь объявляйте класс OutputModule

# тензор x в программе не менять
batch_size = 7
seq_length = 89
in_features = 3
x = torch.rand(batch_size, seq_length, in_features)


# здесь продолжайте программу
class OutputModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(24, 5)

    def forward(self, x):
        y, _ = x

        batch_size, n, _ = y.size()

        out = torch.empty(batch_size, n, self.layer.out_features)

        for i in range(n):
            out[:, i, :] = self.layer(y[:, i, :])
        return out


# тензор x в программе не менять
batch_size = 7
seq_length = 89
in_features = 3
x = torch.rand(batch_size, seq_length, in_features)

# здесь продолжайте программу
model = nn.Sequential(
    nn.LSTM(in_features, 12, batch_first=True, bidirectional=True),
    OutputModule(),
)

model.eval()

out = model(x)
```