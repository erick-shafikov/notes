```python
import numpy as np


def func(x):
    return 0.5 * x ** 2 - 0.1 / np.exp(-x) + 0.5 * np.cos(2 * x) - 2.


def model(x, w):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * np.cos(2 * x) + w[4] * np.sin(2 * x)


def loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def gradients(x, y_true, w):
    y_pred = model(x, w)
    error = y_pred - y_true
    return 2 * error * np.array([1, x, x ** 2, np.cos(2 * x), np.sin(2 * x)])


coord_x = np.arange(-5.0, 5.0, 0.1)  # Значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x)  # Значения функции по оси ординат

sz = len(coord_x)  # Количество значений функций (точек)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01])  # Разные шаги обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.])  # Начальные значения параметров модели
N = 500  # Число итераций градиентного алгоритма SGD
lm = 0.02  # Значение параметра лямбда для вычисления скользящего экспоненциального среднего
Qe = np.mean(loss(coord_y, model(coord_x, w)))  # начальное значение скользящего среднего эмпирического риска
np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

# Стохастический градиентный спуск
for i in range(N):
    k = np.random.randint(sz - 1)  # выбор случайного индекса
    x_k = coord_x[k]
    y_k = coord_y[k]

    # Вычисление градиентов
    grads = gradients(x_k, y_k, w)

    # Обновление весов с учетом разных шагов обучения для каждого параметра
    w -= eta * grads

    # Обновление скользящего среднего эмпирического риска
    Qe = lm * loss(y_k, model(x_k, w)) + (1 - lm) * Qe

# Финальная ошибка после всех итераций по всей выборке
Q = np.mean(loss(coord_y, model(coord_x, w)))

```