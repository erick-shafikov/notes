# Sequential

Принимает другие модули

Методы:

- append

```python
import torch.nn as nn

n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))
n.append(nn.Linear(3, 4))
```

- extend

```python
import torch.nn as nn

n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))
other = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 5))
n.extend(other)  # or `n + other`
```

- insert

```python
import torch.nn as nn

n = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3))
n.insert(0, nn.Linear(3, 4))
```