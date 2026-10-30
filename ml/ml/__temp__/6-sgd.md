# Стохастический градиентный спуск

## 1. Мини-батчи и оценка полного риска

```python
# Условие:
# Задача на повторение / аналог предыдущей.
#
# Повторение SGD: приблизить f(x)=0.5*x+0.2*x**2-0.05*x**3+0.2*sin(4*x)-2.5 на [-4,6] полиномом
# третьей степени. Псевдоградиент усреднять по непрерывному случайному мини-батчу длины
# batch_size, начиная с k; использовать квадратичную потерю. Обновлять Qe по риску батча;
# сохранить w, Q (MSE по всей выборке), Qe. Настройки из заготовки; не печатать.

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

## 2. Повторение: полином четвёртой степени

```python
# Условие:
# Задача на повторение / аналог предыдущей.
#
# Повторение SGD: приблизить f(x)=0.02*exp(-x)-0.2*sin(3*x)+0.5*cos(2*x)-7 на [-5,5] полиномом
# четвёртой степени. На каждом шаге выбирать один случайный отсчёт и обновлять веса по
# квадратичной потере, а также Qe. Итоги: w, MSE на всей выборке в Q, последнее Qe. Использовать
# исходные параметры; без вывода.

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
for i in range(N):
    k = np.random.randint(0, sz-1)
    lossK = loss(w, x[k],y[k])
    w -= eta * df(w, x[k], y[k])
    Qe = lm * lossK + (1-lm) * Qe
Q = np.mean(loss(w, x, y))
```
