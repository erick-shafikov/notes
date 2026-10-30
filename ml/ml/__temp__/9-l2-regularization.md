# L2-регуляризация

## 1. Аналитическая ridge-регрессия

```python
# Условие:
# Аппроксимировать f(x)=0.1*x+0.1*x**2-0.5*sin(2*x)+cos(4*x)+10 на [-3,4] полиномом с 22
# коэффициентами. По X_train,y_train найти w=(X_train^T X_train+lambda*I)^(-1)X_train^T y_train,
# исключив свободный член из штрафа (I[0,0]=0). Среднюю квадратичную ошибку Q посчитать на всех
# исходных отсчётах, не только обучающих. Сохранить w и Q, без вывода.

import numpy as np


def func(x):
    return 0.1 * x + 0.1 * x ** 2 - 0.5 * np.sin(2 * x) + 1 * np.cos(4 * x) + 10


x = np.arange(-3.0, 4.1, 0.1)  # значения по оси абсцисс (Ox) с шагом 0,1
y = np.array(func(x))  # значения функции по оси ординат

N = 22  # размер признакового пространства (степень полинома N-1)
lm = 20  # параметр лямбда для L2-регуляризатора

X = np.array([[a ** n for n in range(N)] for a in x])  # матрица входных векторов
IL = np.array([[lm if i == j else 0 for j in range(N)] for i in range(N)])  # матрица lambda*I
IL[0][0] = 0  # первый коэффициент не регуляризуем

X_train = X[::2]  # обучающая выборка (входы)
Y_train = y[::2]  # обучающая выборка (целевые значения)

w = np.linalg.inv(X_train.T @ X_train + IL) @ X_train.T @ Y_train

Q = np.mean(np.square(X @ w - y))
```

## 2. Повторение ridge-регрессии в уроке SVD

```python
# Условие:
# Задача на повторение / аналог предыдущей.
#
# Повторение ridge-регрессии: приблизить f(x)=0.1*x+0.1*x**2-0.5*sin(2*x)+cos(4*x)+10 на [-3,4]
# полиномом с 22 коэффициентами. Аналитически найти w по X_train,y_train с L2-штрафом и
# I[0,0]=0. MSE Q считать по всей сетке x, не только обучающей. Сохранить w,Q; не печатать.

import numpy as np


def func(x):
    return 0.1 * x + 0.1 * x ** 2 - 0.5 * np.sin(2*x) + 1 * np.cos(4*x) + 10


x = np.arange(-3.0, 4.1, 0.1) # значения по оси абсцисс (Ox) с шагом 0,1
y = np.array(func(x)) # значения функции по оси ординат

N = 22  # размер признакового пространства (степень полинома N-1)
lm = 20  # параметр лямбда для L2-регуляризатора
X_func = lambda x : np.array([x**n for n in range(N)])
X = np.array([[a ** n for n in range(N)] for a in x])  # матрица входных векторов
IL = np.array([[lm if i == j else 0 for j in range(N)] for i in range(N)])  # матрица lambda*I
IL[0][0] = 0  # первый коэффициент не регуляризуем

X_train = X[::2]  # обучающая выборка (входы)
Y_train = y[::2]  # обучающая выборка (целевые значения)

w = np.linalg.inv(X_train.T @ X_train + IL) @ X_train.T @ Y_train
model = lambda x,w : np.dot(x,w)
Q = 0 
for i in range(len(y)):
    xi = X_func(x[i])
    Q += (y[i] - model(xi,w))**2
Q = Q / len(y)
```
