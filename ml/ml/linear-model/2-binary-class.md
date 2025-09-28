- вычисление отступов

```python
import numpy as np

w = np.array([15 / 7, -9 / 7, -1])
x_test = np.array(
    [[1, -8, -4], [1, -2, 2], [1, 4, 8], [1, 6, 3]])  # задайте самостоятельно (признаки образов: x0, x1, x2)
y_test = np.array([1, 1, -1, -1])  # задайте самостоятельно (метки класса)

# здесь продолжайте программу

margin = list(np.dot(w, x) * y_test[i] for i, x in enumerate(x_test))
```

```python
import numpy as np

x_test = np.array([(-5, 2), (-4, 6), (3, 2), (3, -3), (5, 5), (5, 2), (-1, 3)])
y_test = np.array([1, 1, 1, -1, -1, -1, -1])
w = np.array([-8 / 3, -2 / 3, 1])

# здесь продолжайте программу
# массив вида [1, x1, x2] из data_x
X = np.column_stack((np.ones(len(x_test)), x_test))
print(np.sum(y_test * (X @ w) < 0))
```

# бинарная классификация, алгоритм Розенблата

```python

import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

n_train = len(x_train)                          # размер обучающей выборки
w = [0, -1]                                     # начальное значение вектора w
a = lambda x: np.sign(x[0]*w[0] + x[1]*w[1])    # решающее правило
N = 50                                          # максимальное число итераций
L = 0.1                                         # шаг изменения веса
e = 0.1                                         # небольшая добавка для w0 чтобы был зазор между разделяющей линией и граничным образом

last_error_index = -1                           # индекс последнего ошибочного наблюдения

for n in range(N):
    for i in range(n_train):                # перебор по наблюдениям
        if y_train[i]*a(x_train[i]) < 0:    # если ошибка классификации,
            w[0] = w[0] + L * y_train[i]    # то корректировка веса w0
            last_error_index = i

    Q = sum([1 for i in range(n_train) if y_train[i]*a(x_train[i]) < 0])
    if Q == 0:      # показатель качества классификации (число ошибок)
        break       # останов, если все верно классифицируем

if last_error_index > -1:
    w[0] = w[0] + e * y_train[last_error_index]

print(w)

```
