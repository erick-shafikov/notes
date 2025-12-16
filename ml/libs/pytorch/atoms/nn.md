# Нейронная сеть

Двухслойная нейронная сеть с сигмоидной функцией активации для скрытого слоя и линейной функцией активации у выходного
нейрона:

```python
import torch

# значения списков w и g в программе не менять
w = list(map(float, input().split()))
g = list(map(float, input().split()))

W = torch.tensor(w).resize_(2, -1)
W1 = W[:, 1:]
bias1 = W[:, 0]

G = torch.tensor(g)
W2 = G[1:]
bias2 = G[0]

t_inp = torch.rand(3) * 10

u = torch.sigmoid(W1 @ t_inp + bias1)
y = W2 @ u + bias2
```
