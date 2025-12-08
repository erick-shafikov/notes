# Квадратичная функция потерь

```python
import torch

# значения x, func, predict не менять
x = torch.arange(-3, 3, 0.1)
func = x ** 2 - 2 * torch.cos(x) - 5
predict = func + torch.empty_like(func).normal_(0, 0.5)

loss_func = torch.nn.MSELoss()
Q = loss_func(predict, func)
Q_mse = torch.mean((predict - func) ** 2)
```

# Логарифмическая функция потерь

```python
import torch

# значения predict, target не менять
batch_size = 8
target = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32)  # целевые значения
predict = torch.empty(batch_size, 1).normal_(0, 2.0)  # прогнозные значения

loss_func = torch.nn.BCEWithLogitsLoss()
Q = loss_func(predict, target)

p = torch.nn.functional.sigmoid(predict)
Q_bce = -1 * torch.mean(target * torch.log(p) + (1 - target) * torch.log(1 - p))
```