# dataloader

Могут использовать:

- Map-style datasets с __getitem__() и __len__()
- Iterable-style datasets - для динамических данных

```python
from torch.utils.data import DataLoader, Dataset

dataset = Dataset()

DataLoader(
    dataset,  # объект Dataset
    batch_size=1,  # количество семплов
    shuffle=False,  # перемешивание
    sampler=None,  # объект Sampler или Iterable
    batch_sampler=None,  #
    num_workers=0,  # количество процессов для загрузки
    collate_fn=None,  #
    pin_memory=False,  #
    drop_last=False,  # сброс последних непопавших экзепляров dataset
    timeout=0,  #
    worker_init_fn=None,  #
    prefetch_factor=2,  #
    persistent_workers=False,  #
)
```

- dataset объект [Dataset](dataset.md)