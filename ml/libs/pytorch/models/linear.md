# Linear

Наследуется от [Module](./module.md)

```python
from torch import nn

lin = nn.Linear(
    in_features=3,  # размер входного семпла
    out_features=3,  # размер выходного семпла
    bias=True,
    device=None,
)
# веса
weights = lin.weight
# веса
bias = lin.bias  # по умолчанию норм. распределены от (-k^1/2 ; k^1/2 ) k = 1/n_features
```
