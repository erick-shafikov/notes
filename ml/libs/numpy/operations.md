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

# np.mean

Среднее арифметическое

```python
import numpy as np

matrix = np.array(
  [
    [1, 2, 3],
    [4, 5, 6]
  ]
)

print(matrix.mean())  # (1+2+3+4+5+6)/6 = 3.5
```

# np.polyval

- для работы с полиномами

```python
import numpy as np

# Коэффициенты полинома: [a_n, a_{n-1}, ..., a_1, a_0] для a_n*x^n + ... + a_0
coefficients = [2, -3, 1]  # 2x² - 3x + 1

# Точки x, в которых нужно вычислить полином
x_values = [0, 1, 2, 3]

# Вычисление значений полинома
y_values = np.polyval(coefficients, x_values)
y_1 = np.polyval(coefficients, 1)

print(y_values)  # Вывод: [1., 0., 3., 10.]
print(y_1)  # Вывод: 0 значение от 1
```

# np.power

- возведение в степень

```python
import numpy as np

print(np.power(2, 3))  # 2³ = 8

# Возведение массива в степень
arr = np.array([1, 2, 3])
print(np.power(arr, 2))  # [1², 2², 3²] = [1, 4, 9]

# Вычисление значений полинома


a = np.array([2, 3, 4])
b = np.array([3, 2, 1])

np.power(a, b) #[8, 9, 4]

np.power(2, -2)# Ошибка

a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([2, 3])

np.power(a, b)
# array([
#  [ 1,  8],
#  [ 9, 64]
# ])
# Поскольку b растягивается до:
# [
#  [2, 3],
#  [2, 3]
# ]
```

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

# np.square

```python
import numpy as np

# Для скаляра
print(np.square(5))  # 25

# Для одномерного массива
arr = np.array([1, 2, 3, 4])
print(np.square(arr))  # [1, 4, 9, 16]
```

# np.vstack

```python
import numpy as np

a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([
    [5, 6]
])

result = np.vstack([a, b])
# [
#  [1 2]
#  [3 4]
#  [5 6]
# ]

```
