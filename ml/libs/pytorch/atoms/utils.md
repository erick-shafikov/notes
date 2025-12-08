# cat

Объединяет тензоры вдоль указанной оси

```python
import torch

a = torch.tensor([[1, 2]])
b = torch.tensor([[3, 4]])
c = torch.tensor([[5, 6]])

result = torch.cat((a, b, c), dim=0)  # tensor([[1, 2], [3, 4], [5, 6]])
result = torch.cat((a, b), dim=1)  # tensor([[1, 2, 3, 4]])

```

# vstack

Склеивает векторы по 0 оси, то есть torch.vstack() == torch.cat(..., dim=0)

```python
import torch

a = torch.tensor([[1, 2]])
b = torch.tensor([[3, 4]])
c = torch.tensor([[5, 6]])

result = torch.vstack((a, b, c))  # tensor([[1, 2],[3, 4],[5, 6]])
```

# stack

Создает новую ось из векторов одинаковых размеров

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

res = torch.stack([a, b])  # tensor([[1, 2, 3], [4, 5, 6]])
```