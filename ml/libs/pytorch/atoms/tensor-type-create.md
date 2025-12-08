```python
import numpy as np
import torch
from numpy.conftest import dtype

# тензоры должны быть равных размерностей

# создание пустого 2 * 5 * 2 тензора
t = torch.empty(3, 5, 2)
# все будет float
torch.tensor([1, 2.0, 0])
# явное указание
torch.tensor([1, 2, 0], dtype=torch.float32)
# типизированный тензор
torch.ByteTensor([1, 2])
torch.DoubleTensor([1, 2, 3])
# torch.FloatTensor 32 бита, с плавающей точкой
# torch.DoubleTensor 64 бита, с плавающей точкой
# torch.IntTensor 32 бита, целочисленный, знаковый
# torch.LongTensor 64 бита, целочисленный, знаковый
# torch.BoolTensor булевый (True/False)
```

# Свойства и атрибуты тензоров

```python
import torch

t = torch.empty(3, 5, 2)

# атрибут, содержащий тип данных тензора
t.dtype
# атрибут, содержащий размерность тензора
t.shape

# методы
# метод, возвращающий класс тензора (FloatTensor, LongTensor и т.п.)
t.type()
# метод, возвращающий размерность тензора
t.size()
# метод, возвращающий число осей тензора
t.dim()
# число осей
torch.shape()
# изменения типов данных на лету
t = t.float()  # half, float, double, short
# float() == torch.float32
# double() == torch.float64
# int() == torch.int32
# long() == torch.int64
# char() == torch.int8
# byte() == torch.uint8
# bool() == torch.bool
```

# Взаимодействие с np

```python
import torch
import numpy as np

# взаимодействие с np
# преобразование из np.array
d_np = np.array([1, 2, 3])
# не копирует, создает с той же размерностью
t2 = torch.from_numpy(d_np)
# копирует значения, но списки копируются
t3 = torch.tensor(d_np)
# обратное преобразование (но не копия, а ссылка)
d_linked = t2.numpy()
# обратное преобразование (с копией)
d_unlinked = t2.numpy().copy()
```