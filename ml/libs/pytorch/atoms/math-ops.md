```python
import torch

a13 = torch.tensor([1, 2, 3])
# поэлементные операции
print(a13 - 3)
# вещественный + целочисленный = вещественный

a23 = torch.arange(1, 7).view(2, 3)
# не будет проблемой сложить
# каждый ряд а23 сложится с а13
# должны быть согласованны
print(a23 + a13)

#
a_int = torch.IntTensor([1, 2, 3, 4])
b_float = torch.ones(4)
# ок
b_float += a_int
# ошибка
# a_int += b_float
# мат операции
x = torch.arange(1, 3)
y = torch.arange(4, 6)

res1 = x.add(y)
# inplace метод вернет в x, вернет тот же тензор a
x.add_(y)
res2 = x.sub(y)
res3 = x.mul(y)

```