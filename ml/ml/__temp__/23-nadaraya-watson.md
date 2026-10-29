# Регрессия Надарая — Ватсона

## 1. Интерполяция четырёх точек

По четырём исходным точкам x,y восстановить значения на сетке x_est формулой Надарая — Ватсона: сумма y_i*K(distance/h), делённая на сумму весов. Взять h=1, distance=abs(x-x_i), треугольное ядро K(r)=abs(1-r) при abs(r)<=1, иначе 0. Результат сохранить в y_est. Самостоятельно построить график и сравнить ширины окна и ядра; не печатать.

### Исходные данные и заготовка

```python
import numpy as np

# координаты четырех точек
x = np.array([0, 1, 2, 3])
y = np.array([0.5, 0.8, 0.6, 0.2])

x_est = np.arange(0, 3.1, 0.1) # множество точек для промежуточного восстановления функции
```

### Решение

```python
import numpy as np

# координаты четырех точек
x = np.array([0, 1, 2, 3])
y = np.array([0.5, 0.8, 0.6, 0.2])

x_est = np.arange(0, 3.1, 0.1) # множество точек для промежуточного восстановления функции

h = 1
ro = abs(x_est.reshape(-1, 1) - x)
r = ro / h
w = abs(1 - r) * (r <= 1)
y_est = w @ y / w.sum(1)
```

## 2. Гауссовское сглаживание шумной функции

Для шумной функции f(x)=0.1*x-cos(x/2)+0.4*sin(3*x)+5+noise на сетке [-5,5] с шагом 0.1 выполнить ядерное сглаживание Надарая — Ватсона. Использовать abs-разность, нормальное ядро exp(-r**2/2)/sqrt(2*pi), h=0.5; шум генерируется заготовкой. Сохранить сглаженные значения в y_est и MSE относительно наблюдений y в Q. Самостоятельно сравнить окна и h; без вывода.

### Исходные данные и заготовка

```python
import numpy as np

def func(x):
    return 0.1 * x - np.cos(x/2) + 0.4 * np.sin(3*x) + 5

np.random.seed(0)

x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
y = func(x) + np.random.normal(0, 0.2, len(x)) # значения функции по оси ординат
```

### Решение

```python
import numpy as np


def func(x):
    return 0.1 * x - np.cos(x/2) + 0.4 * np.sin(3*x) + 5

np.random.seed(0)

x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
y = func(x) + np.random.normal(0, 0.2, len(x)) # значения функции по оси ординат

x_est = np.arange(-5.0, 5.0, 0.1) # множество точек для промежуточного восстановления функции

# Функции

# Расстояние для случая с одни признаком
p = lambda x, y: abs(x - y)

# Ядро (Гаусс)
K = lambda r: 1 / (2 * np.pi)**.5 * np.exp(- r**2 / 2)

# Ядро (треугольное окно Парзена)
K_triangle = lambda r: np.abs(1 - r) * (r <= 1)

# Формула Надарая-Ватсона для прогноза по гаусовскому ядру
a = lambda xi, y, h=1: np.sum([yi * K(p(xi, xx) / h) for xx, yi in zip(x, y)]) / np.sum([K(p(xi, xx) / h) for xx in x])

# Формула Надарая-Ватсона для прогноза по треугольному ядру
a_triangle = lambda xi, y, h=1: np.sum([yi * K_triangle(p(xi, xx) / h) for xx, yi in zip(x, y)]) / np.sum([K_triangle(p(xi, xx) / h) for xx in x])

# Предсказание
y_est = [a(xi, y, h=.5) for xi in x_est]
y_est_triangle = [a_triangle(xi, y, h=.5) for xi in x_est]

# Качество
Q = np.mean((y_est - y)**2)
Q_triangle = np.mean((y_est_triangle - y)**2)
# print('Q = ', Q, 'Q_triangle = ', Q_triangle,)

# График
import matplotlib.pyplot as plt

plt.scatter(x, y, color='red', label='Исходные точки', marker = '.')
plt.plot(x_est, func(x), '--', color='red', label='Функция без дисперсии')
plt.plot(x_est, y_est, color='blue', label='Прогноз с гауссовскиим ядром')
plt.plot(x_est, y_est_triangle, color='green', label='Прогноз с треугольным ядром')
plt.title('Интерполяция с использованием метода Надарая-Ватсона')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
```

## 3. Последовательный прогноз временного ряда

Для заданного ряда rub_usd последовательно спрогнозировать следующие 10 дней формулой Надарая — Ватсона. Ядро гауссовское, расстояние — абсолютная разность индексов дней, h=3. Каждый новый прогноз включать в историю перед следующим шагом. Сохранить только 10 новых значений в predict; не печатать.

### Исходные данные и заготовка

```python
import numpy as np

rub_usd = np.array([75, 76, 79, 82, 85, 81, 83, 86, 87, 85, 83, 80, 77, 79, 78, 81, 84])
```

### Решение

```python
import numpy as np

rub_usd = np.array([75, 76, 79, 82, 85, 81, 83, 86, 87, 85, 83, 80, 77, 79, 78, 81, 84])

x = np.array(range(len(rub_usd)))
x_est = np.array(range(len(rub_usd), len(rub_usd) + 10))


h = 3
ro = lambda xi, x: np.abs(xi - x)
K = lambda r: np.exp(-0.5 * r * r) / np.sqrt(2 * np.pi)
w = lambda xx, xi: K(ro(xx, xi) / h) 

predict = []

for xx in x_est:
    ww = np.array([w(xx, xi) for xi in x])
    yy = np.dot(ww, rub_usd) / sum(ww)
    predict.append(yy)
    rub_usd = np.append(rub_usd, yy)[1:]
    x = np.append(x, xx)[1:]
```
