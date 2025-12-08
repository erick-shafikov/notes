# np.mean

Среднее арифметическое

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

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

print(y_values)  # Вывод: [1., 0., 3., 10.]
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


# numpy.power(x1, x2, out=None, where=True, dtype=None)
coord_x = np.arange(-5.0, 5.0, 0.1).reshape(-1, 1)
X = np.power(coord_x, np.arange(5))
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