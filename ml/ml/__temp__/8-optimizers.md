# Оптимизаторы градиентного спуска

## 1. Метод импульсов

```python
# Условие:
# Минимизировать f(x)=-0.5*x+0.2*x**2-0.01*x**3-0.3*sin(4*x) методом импульсов:
# v=gamma*v+(1-gamma)*eta*f′(x), x-=v. Параметры: eta=0.1, x=-3.5, N=200, gamma=0.8, v=0.
# Сохранить x. Для самостоятельной проверки сравнить с обычным спуском; не печатать.

import numpy as np


def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4*x)


# здесь объявляйте функцию df (производную) и продолжайте программу
def df(x):
    return -0.5 + 0.4 * x - 0.03 * x ** 2 - 0.3 * 4 * np.cos(4*x)


eta = 0.1
x = -3.5
N = 200
gama = 0.8
v = 0

for _ in range(N):
    v = gama * v + (1 - gama) * eta * df(x)
    x = x - v
```

## 2. Импульс Нестерова

```python
# Условие:
# Минимизировать f(x)=0.4*x+0.1*sin(2*x)+0.2*cos(3*x) методом Нестерова, вычисляя производную в
# x-gamma*v. Начальные значения: eta=1, x=4, N=500, gamma=0.7, v=0. Обновление
# v=gamma*v+(1-gamma)*eta*f′(x-gamma*v), затем x-=v. Сохранить x; сравнить с обычным спуском
# самостоятельно.

import numpy as np


def f(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)


def dfdx(x):
    return 0.4 + 0.2 * np.cos(2 * x) - 0.6 * np.sin(3 * x)


eta = 1.
x0 = 4.
N = 500
gamma = 0.7
nu = 0

for _ in range(N):
    nu = gamma * nu + (1 - gamma) * eta * dfdx(x0 - gamma * nu)
    x0 -= nu

x = x0
```

## 3. RMSProp для одной переменной

```python
# Условие:
# Минимизировать f(x)=2*x+0.1*x**3+2*cos(3*x) алгоритмом RMSProp. Начало x=4, eta=0.5, N=200,
# alpha=0.8, G=0, epsilon=0.01. Обновлять G=alpha*G+(1-alpha)*f′(x)**2 и
# x-=eta*f′(x)/(sqrt(G)+epsilon). Итог в x, без вывода.

import numpy as np


def f(x):
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3 * x)


def dfdx(x):
    return 2 + 0.3 * x ** 2 - 6 * np.sin(3 * x)


eta = .5
x0 = 4.
N = 200
alpha = .8
G = 0
epsilon = .01

for _ in range(N):
    G = alpha * G + (1 - alpha) * dfdx(x0) ** 2
    x0 -= eta * dfdx(x0) / (np.sqrt(G) + epsilon)

x = x0
```

## 4. Повторение метода импульсов в уроке SVD

```python
# Условие:
# Задача на повторение / аналог предыдущей.
#
# Повторить минимизацию f(x)=-0.5*x+0.2*x**2-0.01*x**3-0.3*sin(4*x) с импульсом. Начало x=-3.5,
# eta=0.1, N=200, gamma=0.8, v=0. На шаге v=gamma*v+(1-gamma)*eta*f′(x), затем x-=v. Сохранить
# x; самостоятельно сравнить с обычным спуском.

import numpy as np
import matplotlib.pyplot as plt

# Функции
func = lambda x: -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4*x)
df = lambda x: -0.5 + 0.4 * x - 0.03 * x**2  - 1.2 * np.cos(4*x)

# Параметры
rho = 0.1 
x = -3.5
x_ordinary = -3.5
N = 200 # число итераций градиентного алгоритма 
gamma = 0.8
v = 0 # начальное значение для v

# Список для хранения значений x на каждой итерации
x_values = []

for _ in range(N):
    v = gamma * v + (1 - gamma) * rho * df(x)
    x = x - v
    x_ordinary -= rho * df(x_ordinary)
    # x_values.append(x)

# Создание графика
x_range = np.linspace(-4, 4, 400)
y_range = func(x_range)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, label='Функция', color='blue')
# plt.scatter(x_values[-1], func(x_values[-1]), color='red', label='Минимум', zorder=5)
plt.scatter(x, func(x), color='red', label='Минимум с импульсом', zorder=5)
plt.scatter(x_ordinary, func(x_ordinary), color='green', label='Минимум без импульса', zorder=5)
plt.title('График функции и её минимумы')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()
```
