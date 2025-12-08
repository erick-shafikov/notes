# GRU

```python
import torch
import torch.nn as nn

rnn = nn.GRU(
    10,
    20,
    batch_first=True,
    bidirectional=True,
    dropout=0.4,  # применится ко всем полносвязанным, кроме последнего
)

x = torch.randn(7, 3, 10)
y, h = rnn(x)

print(y.size())  # torch.Size([7, 3, 20]) | torch.Size([7, 3, 40])
print(h.size())  # torch.Size([1, 7, 20]) | torch.Size([2, 7, 20])

```