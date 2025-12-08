# Формирование данных

- сформировать массив вида [[1,1], [1, 2], [1, 3] ... [1, n]]

```python
import numpy as np

x = np.arange(-1.0, 1.0, 0.1)
np.stack([np.ones_like(x), x], axis=1)
# OR
ones = np.ones((len(x), 1))
X = np.hstack((ones, x.reshape(-1, 1)))
```

- добавить число в каждую строку массива

```python
import numpy as np

x_test = np.array([(-5, 2), (-4, 6), (3, 2), (3, -3), (5, 5), (5, 2), (-1, 3)])
X = np.column_stack((np.ones(len(x_test)), x_test))
# x_test = np.array([(1, -5, 2), (1,-4, 6,1), (1,3, 2), (1,3, -3), (1,5, 5), (1,5, 2), (1,-1, 3)])
```

# Вычисление составных частей

- линейная модель

```python
#  a(x, w) = w0 + w1 * x
model_a = lambda m_x, m_w: (m_w[1] * m_x + m_w[0])
```

- оценка качества

```python
import numpy as np

x_train = np.array([])
y_train = np.array([])
w = []

Q = np.mean((x_train @ w - y_train) ** 2)
```

- квадратичная функция потерь

```python
# loss(a, y) = (a(x, w) - yi) ** 2
loss = lambda ax, y: (ax - y) ** 2
```

- Вычислите показатель качества (количество ошибочных классификаций):

```python

import numpy as np

x_test = np.array([])
X = np.column_stack((np.ones(len(x_test)), x_test))
y_test = np.array([])
w = np.array([])
# E(yi * a(xi) < 0)
np.sum(y_test * (X @ w) < 0)


```

- Байесовский гауссовский классификатор

```python
import numpy as np

ax = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))
```