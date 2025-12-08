# unflatten

Расширяет тензор до нужной размерности, два параметра:

- dim - размерность, которая будет расширена
- unflattened_size - форма в которую превратится размерность dim

```python
from torch import nn
import torch

input = torch.randn(2, 50)
# With tuple of ints
m = nn.Sequential(
    nn.Linear(50, 50),
    nn.Unflatten(1, (2, 5, 5))
)
output = m(input)
output.size()
# With torch.Size
m = nn.Sequential(
    nn.Linear(50, 50),
    nn.Unflatten(1, torch.Size([2, 5, 5]))
)
output = m(input)
output.size()
# With namedshape (tuple of tuples)
input = torch.randn(2, 50, names=("N", "features"))
unflatten = nn.Unflatten("features", (("C", 2), ("H", 5), ("W", 5)))
output = unflatten(input)
output.size()
```