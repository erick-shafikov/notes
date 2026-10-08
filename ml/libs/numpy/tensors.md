# np.reshape

Позволяет преобразовать размерность массива
В np.reshape() можно использовать одно отрицательное значение -1, чтобы NumPy сам вычислил нужный размер этой оси.
Можно использовать только один -1:

```python
a = np.arange(6).reshape((3, 2))
# [[0, 1],
# [2, 3],
# [4, 5]]

a = np.arange(12)

a.reshape(3, -1)
# [
#  [ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
# ]
a.reshape(-1, 2)
# [
#  [ 0  1]
#  [ 2  3]
#  [ 4  5]
#  [ 6  7]
#  [ 8  9]
#  [10 11]
# ]
a.reshape(-1, 1)
# [
#  [ 0]
#  [ 1]
#  [ 2]
#  ...
#  [11]
# ]
```

# np.stack

```python
# сформировать массив вида [[1,1], [1, 2], [1, 3] ... [1, n]]
import numpy as np

np.stack([np.ones_like(x), x], axis=1)
```

## np.column_stack

Количество строк должно совпадать.

```python
# добавить число в каждую строку массива
import numpy as np

x_test = np.array([(-5, 2), (-4, 6), (3, 2), (3, -3), (5, 5), (5, 2), (-1, 3)])
X = np.column_stack((np.ones(len(x_test)), x_test))
# x_test = np.array([(1, -5, 2), (1,-4, 6,1), (1,3, 2), (1,3, -3), (1,5, 5), (1,5, 2), (1,-1, 3)])
```

```python
x = np.array([1,2,3])

X = np.column_stack((
    np.ones(len(x)),
    x,
    x**2
))

# [1, x, x²]
# [
#  [1,1,1],
#  [1,2,4],
#  [1,3,9]
# ]
```

# np.vstack

```python
import numpy as np

a = np.array([[1, 2],[3, 4]])
b = np.array([[5, 6]])

result = np.vstack([a, b])
# [[1 2]
#  [3 4]
#  [5 6]]

```

# np.hstack

Горизонтальное объединение

```python
import numpy as np

a = np.array([
    [1],
    [2],
    [3]
])

b = np.array([
    [10],
    [20],
    [30]
])

result = np.hstack([a, b])
# [
#  [ 1 10]
#  [ 2 20]
#  [ 3 30]
# ]
```

```python
# сформировать массив вида [[1,1], [1, 2], [1, 3] ... [1, n]]
x = np.arange(-1.0, 1.0, 0.1)
ones = np.ones((len(x), 1))
X = np.hstack((ones, x.reshape(-1, 1)))
```
