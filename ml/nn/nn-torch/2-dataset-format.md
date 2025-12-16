# Структура папок

- dataset_reg
  - train
    - file1.ext
    - file2.ext
    - format.json
  - test
    - file1.ext
    - file2.ext
    - format.json

# Dataset

Класс, который формирует выборку и отдает данные. Позволит организовать загрузку данных для выборки

[Dataset](../../libs/pytorch/models/dataset.md)

```python
# Dataset - универсальное представление обучающих и тестовых данных
# экономия памяти с большими выборками
from torch.utils import data


class MyDataset(data.Dataset):
    def __init__(self):  # инициализация переменных объекта
        pass

    def __getitem__(self, item):  # возвращение образа выборки по индексу item
        pass

    def __len__(self):  # возвращение размера выборки
        pass


d_train = MyDataset()
j = 1
X_j, y_j = d_train[j]
data_size = len(d_train)
```

# Dataloader

Итератор для извлечения данных

[Dataloader](../pytorch/models/dataloader.md)

```python
from torch.utils import data

# Dataloader помогает перемещать объекты выборки
# drop_last - если размер выборки не кратен размеру пакетов
d_train = data.Dataset()
train_data = data.DataLoader(d_train, batch_size=4, shuffle=True, drop_last=False)
# Dataloader имеет метод __iter__ и __next__
# эти методы обращаются к __getitem__ у Dataset

for x_train, y_train in train_data:
    # можно перебирать входные данные и целевое значение
    pass

```

# Формирование FuncDataset и использование DataLoader

```python
import torch
import torch.utils.data as data

data_x = [(5.3, 2.3), (5.7, 2.5), (4.0, 1.0), (5.6, 2.4), (4.5, 1.5), (5.4, 2.3), (4.8, 1.8), (4.5, 1.5), (5.1, 1.5),
          (6.1, 2.3), (5.1, 1.9), (4.0, 1.2), (5.2, 2.0), (3.9, 1.4), (4.2, 1.2), (4.7, 1.5), (4.8, 1.8), (3.6, 1.3),
          (4.6, 1.4), (4.5, 1.7), (3.0, 1.1), (4.3, 1.3), (4.5, 1.3), (5.5, 2.1), (3.5, 1.0), (5.6, 2.2), (4.2, 1.5),
          (5.8, 1.8), (5.5, 1.8), (5.7, 2.3), (6.4, 2.0), (5.0, 1.7), (6.7, 2.0), (4.0, 1.3), (4.4, 1.4), (4.5, 1.5),
          (5.6, 2.4), (5.8, 1.6), (4.6, 1.3), (4.1, 1.3), (5.1, 2.3), (5.2, 2.3), (5.6, 1.4), (5.1, 1.8), (4.9, 1.5),
          (6.7, 2.2), (4.4, 1.3), (3.9, 1.1), (6.3, 1.8), (6.0, 1.8), (4.5, 1.6), (6.6, 2.1), (4.1, 1.3), (4.5, 1.5),
          (6.1, 2.5), (4.1, 1.0), (4.4, 1.2), (5.4, 2.1), (5.0, 1.5), (5.0, 2.0), (4.9, 1.5), (5.9, 2.1), (4.3, 1.3),
          (4.0, 1.3), (4.9, 2.0), (4.9, 1.8), (4.0, 1.3), (5.5, 1.8), (3.7, 1.0), (6.9, 2.3), (5.7, 2.1), (5.3, 1.9),
          (4.4, 1.4), (5.6, 1.8), (3.3, 1.0), (4.8, 1.8), (6.0, 2.5), (5.9, 2.3), (4.9, 1.8), (3.3, 1.0), (3.9, 1.2),
          (5.6, 2.1), (5.8, 2.2), (3.8, 1.1), (3.5, 1.0), (4.5, 1.5), (5.1, 1.9), (4.7, 1.4), (5.1, 1.6), (5.1, 2.0),
          (4.8, 1.4), (5.0, 1.9), (5.1, 2.4), (4.6, 1.5), (6.1, 1.9), (4.7, 1.6), (4.7, 1.4), (4.7, 1.2), (4.2, 1.3),
          (4.2, 1.3)]
data_y = [1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
          -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1,
          -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1,
          1, -1, -1, -1, -1, -1]


class FuncDataset(data.Dataset):
    def __init__(self, data_x, data_y):
        self.data = data_x
        self.targets = data_y
        self.length = len(data_x)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return self.length


batch_size = 12
d_train = FuncDataset(torch.tensor(data_x), torch.tensor(data_y))
train_data = data.DataLoader(d_train, batch_size, shuffle=True, drop_last=True)

for x, t in train_data:
    print(x, t)
```
