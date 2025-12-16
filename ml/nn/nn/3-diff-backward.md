# backward

Позволяет найти производную от заданных значений в точке

```python
import torch

x0, x1, x2, x3 = map(float, input().split())  # переменные x0, x1, x2, x3 в программе не менять

cords_x = torch.arange(-4, 6, 0.1)  # точки интервала [-4; 6) с шагом 0.1 (тензор в программе не менять)

# указываем множественные параметры которые буду подвергнуты дифференцированию
w0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
w1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True)
w2 = torch.tensor(x2, dtype=torch.float32, requires_grad=True)
w3 = torch.tensor(x3, dtype=torch.float32, requires_grad=True)


def predict(x):
    return w0 + w1 * x + w2 * x ** 2 + w3 * x ** 3


def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * torch.cos(3 * x) + 2


# сама функция,
L = torch.mean((predict(cords_x) - func(cords_x)) ** 2)

L.backward()

```