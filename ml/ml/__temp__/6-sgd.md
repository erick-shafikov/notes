# Стохастический градиентный спуск

## 1. SGD и скользящая оценка риска

Аппроксимировать f(x)=0.5*x**2-0.1/exp(-x)+0.5*cos(2*x)-2 на [-5,5] моделью с признаками [1,x,x**2,cos(2*x),sin(2*x)]. На каждом шаге SGD выбирать один случайный объект, обновлять веса и Qe=lambda*L_k+(1-lambda)*Qe. Сохранить w, итоговый MSE в Q и последнее Qe. Параметры цикла заданы в заготовке; вывод не нужен.

### Решение

```python
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.animation import FFMpegWriter


def func(x):
    return 0.5 * x ** 2 - 0.1 * 1 / np.exp(-x) + 0.5 * np.cos(2 * x) - 2.


def model(x, w):
    w0, w1, w2, w3, w4 = w
    return w0 + w1 * x + w2 * x ** 2 + w3 * np.cos(2 * x) + w4 * np.sin(2 * x)


def loss_i(coord_x, coord_y, w, i):
    return (model(coord_x[i], w) - coord_y[i]) ** 2


def gradient_k(coord_x, coord_y, w, k):
    xi = lambda x: np.array([1, x, x ** 2, np.cos(2 * x), np.sin(2 * x)])
    s = xi(coord_x[k])
    grad = (np.dot(w, s) - coord_y[k]) * s
    return 2 * grad


coord_x = np.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)
sz = len(coord_x)

eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01])
w = np.array([0., 0., 0., 0., 0.])
N = 500
lm = 0.02

Qe = np.sum([loss_i(coord_x, coord_y, w, i) for i in range(sz)]) / sz

np.random.seed(0)

for i in range(N):
    k = np.random.randint(0, sz)
    w -= eta * gradient_k(coord_x, coord_y, w, k)
    Qe = lm * loss_i(coord_x, coord_y, w, k) + (1 - lm) * Qe

Q = np.sum([loss_i(coord_x, coord_y, w, i) for i in range(sz)]) / sz
```

## 2. Мини-батчи и оценка полного риска

**Задача на повторение / аналог предыдущей.**

Повторение SGD: приблизить f(x)=0.5*x+0.2*x**2-0.05*x**3+0.2*sin(4*x)-2.5 на [-4,6] полиномом третьей степени. Псевдоградиент усреднять по непрерывному случайному мини-батчу длины batch_size, начиная с k; использовать квадратичную потерю. Обновлять Qe по риску батча; сохранить w, Q (MSE по всей выборке), Qe. Настройки из заготовки; не печатать.

### Решение

```python
import numpy as np

def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5

def model_a(w, x):
    return w @ x


def loss(w, x):
    return (model_a(w, x) - func(x[1]))**2


def grad_loss(w, x):
    return 2 * (model_a(w, x) - func(x[1])) * x.T



coord_x = np.arange(-4.0, 6.0, 0.1) 
coord_y = func(coord_x) 

sz = len(coord_x)	
eta = np.array([0.1, 0.01, 0.001, 0.0001]) 
w = np.array([0., 0., 0., 0.]) 
N = 500 
lm = 0.02 
batch_size = 50 # размер мини-батча (величина K = 50)

X = np.array([[xi**n for n in range(len(w))] for xi in coord_x])
Qe = 1/sz * sum(loss(w, xi) for xi in X)

np.random.seed(0)

for _ in range(N):
    k = np.random.randint(0, sz-batch_size-1)
    Qk = 1/batch_size * sum([loss(w, X[i]) for i in range(k, k + batch_size)])
    grad_Qk = 1/batch_size * np.sum([grad_loss(w, X[i]) for i in range(k, k + batch_size)], axis=0)
    w = w - eta * grad_Qk
    Qe = lm * Qk + (1-lm) * Qe

Q = 1/sz * sum(loss(w, xi) for xi in X)
```

## 3. Повторение: полином четвёртой степени

**Задача на повторение / аналог предыдущей.**

Повторение SGD: приблизить f(x)=0.02*exp(-x)-0.2*sin(3*x)+0.5*cos(2*x)-7 на [-5,5] полиномом четвёртой степени. На каждом шаге выбирать один случайный отсчёт и обновлять веса по квадратичной потере, а также Qe. Итоги: w, MSE на всей выборке в Q, последнее Qe. Использовать исходные параметры; без вывода.

### Исходные данные и заготовка

```python
k = np.random.randint(0, sz-1) # sz - размер выборки (массива coord_x)
```

### Решение

```python
import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.02 * np.exp(-x) - 0.2 * np.sin(3 * x) + 0.5 * np.cos(2 * x) - 7

def df(w, x, y):
    return 2 * (x @ w - y) * x.T

def loss(w, x, y):
    return (x @ w - y)**2

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 1e-3, 1e-4, 1e-5, 1e-6]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего

x = np.column_stack([np.ones(sz), coord_x, coord_x**2, coord_x**3, coord_x**4])
y = np.array(coord_y)
Qe = np.mean(loss(w, x, y))# начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

# здесь продолжайте программу
for i in range(N):
    k = np.random.randint(0, sz-1)
    lossK = loss(w, x[k],y[k])
    w -= eta * df(w, x[k], y[k])
    Qe = lm * lossK + (1-lm) * Qe
Q = np.mean(loss(w, x, y))
```

## 4. Повторение SGD в уроке кластеризации

**Задача на повторение / аналог предыдущей.**

Повторение SGD: на [-5,5] приблизить f(x)=0.5*x**2-0.1/exp(-x)+0.5*cos(2*x)-2 моделью на признаках [1,x,x**2,cos(2*x),sin(2*x)]. На каждом шаге выбирать один случайный объект и обновлять веса по MSE, а также Qe. Сохранить w, итоговый полный MSE в Q и Qe; параметры из заготовки. Не печатать.

### Исходные данные и заготовка

```python
k = np.random.randint(0, sz-1) # sz - размер выборки (массива coord_x)
```

### Решение

```python
import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

def L(w, x, y):
    return (x @ w - y) ** 2

def dL(w, x, y):
    return 2 * ((x @ w - y) * x.T)

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего

X = np.array([[1, x, x ** 2, np.cos(2 * x), np.sin(2 * x)] for x in coord_x])
Y = np.array(coord_y)

Qe = np.mean(L(w, X, Y))
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for _ in range(N):
    k = np.random.randint(0, sz-1)
    w = w - eta * dL(w, X[k], Y[k])
    Qe = lm * L(w, X[k], Y[k]) + (1 - lm) * Qe

Q = np.mean(L(w, X, Y))
```
