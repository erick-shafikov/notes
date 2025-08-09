- вычисление с помощью l2 регуляризатора

```python

import numpy as np

x = np.arange(0, 10.1, 0.1)
y = np.array([a ** 3 - 10 * a ** 2 + 3 * a + 500 for a in x])  # функция в виде полинома x^3 - 10x^2 + 3x + 500
x_train, y_train = x[::2], y[::2]
N = 13  # размер признакового пространства (степень полинома N-1)
L = 20  # при увеличении N увеличивается L (кратно): 12; 0.2   13; 20    15; 5000

X = np.array([[a ** n for n in range(N)] for a in x])  # матрица входных векторов
IL = np.array([[L if i == j else 0 for j in range(N)] for i in range(N)])  # матрица lambda*I
IL[0][0] = 0  # первый коэффициент не регуляризуем
X_train = X[::2]  # обучающая выборка
Y = y_train  # обучающая выборка

# вычисление коэффициентов по формуле w = (XT*X + lambda*I)^-1 * XT * Y
A = np.linalg.inv(X_train.T @ X_train + IL)
w = A @ X_train.T @ Y

Q = np.mean(np.square(X @ w - y))
```

- l2 + sgd

```python
import numpy as np


# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


# модель
def model(w, x):
    xv = np.array([x ** n for n in range(len(w))])
    return w.T @ xv


# функция потерь
def loss(w, x, y):
    return (model(w, x) - y) ** 2


# производная функции потерь
def dL(w, x, y):
    xv = np.array([x ** n for n in range(len(w))])
    return 2 * (model(w, x) - y) * xv


coord_x = np.arange(-4.0, 6.0, 0.1)  # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x)  # значения функции по оси ординат

N = 5  # сложность модели (полином степени N-1)
lm_l2 = 2  # коэффициент лямбда для L2-регуляризатора
sz = len(coord_x)  # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002])  # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.zeros(N)  # начальные нулевые значения параметров модели
n_iter = 500  # число итераций алгоритма SGD
lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20  # размер мини-батча (величина K = 20)

Qe = loss(w, coord_x, coord_y).mean()  # начальное значение среднего эмпирического риска
np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

for _ in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)
    batch = np.arange(k, k + batch_size)
    batch_x = coord_x[batch]
    batch_y = coord_y[batch]

    Qe = lm * loss(w, batch_x, batch_y).mean() + (1 - lm) * Qe

    w1 = np.array(w)
    w1[0] = 0

    w -= eta * (dL(w, batch_x, batch_y).mean(axis=1) + lm_l2 * w1)

Q = loss(w, coord_x, coord_y).mean()

```