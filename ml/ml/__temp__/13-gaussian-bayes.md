# Гауссовский байесовский классификатор

## 1. Двумерный гауссовский классификатор

Реализовать гауссовский байесовский классификатор с полными ковариационными матрицами для двумерных классов. Генерация: класс -1 имеет r=0.7, mu=[1,-2], D=1; класс +1 — r=0.5, mu=[1,3], D=2. Использовать оценки статистик из заготовки. Для всей обучающей выборки сохранить predict и число ошибок Q (не долю). Без вывода.

### Исходные данные и заготовка

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [1, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

# обучающая выборка для байесовского классификатора (стандартный формат)
x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# вычисление оценок математических ожиданий
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

# вычисление ковариационных матриц
a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# параметры для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

b = lambda l, v, x, m, py: l * py * 1/np.sqrt(2 * np.pi * np.linalg.det(v)) * np.e**(-1/2 * (x - m).T @ np.linalg.inv(v) @ (x - m))
```

### Решение

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [1, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

# обучающая выборка для байесовского классификатора (стандартный формат)
x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# вычисление оценок математических ожиданий
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

# вычисление ковариационных матриц
a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# параметры для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

b = lambda l, v, x, m, py: l * py * 1/np.sqrt(2 * np.pi * np.linalg.det(v)) * np.e**(-1/2 * (x - m).T @ np.linalg.inv(v) @ (x - m))

# здесь продолжайте программу
predict = [np.argmax([b(L1, VV1, x, mm1, Py1), b(L2, VV2, x, mm2, Py2)]) * 2 - 1 for x in x_train]

Q = sum(int(predict[i] != y_train[i]) for i in range(len(y_train)))
```

## 2. Трёхмерный гауссовский классификатор

**Задача на повторение / аналог предыдущей.**

Аналог предыдущей задачи для трёх признаков: класс -1 имеет r=0.7, mu=[-1,-2,-1], D=1; класс +1 — r=0.5, mu=[1,2,1], D=2. Оценить по выборке средние и ковариационные матрицы каждого класса. При равных штрафах и априорных вероятностях реализовать гауссовскую классификацию. Сохранить predict в исходном порядке и количество ошибок Q. Не печатать.

### Исходные данные и заготовка

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [-1, -2, -1]
V1 = [[D1, D1 * r1, D1*r1*r1], [D1 * r1, D1, D1*r1], [D1*r1*r1, D1*r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [1, 2, 1]
V2 = [[D2, D2 * r2, D2*r2*r2], [D2 * r2, D2, D2*r2], [D2*r2*r2, D2*r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])
```

### Решение

```python
import numpy as np

np.random.seed(0)
# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [-1, -2, -1]
V1 = [[D1, D1 * r1, D1*r1*r1], [D1 * r1, D1, D1*r1], [D1*r1*r1, D1*r1, D1]]
r2 = 0.5
D2 = 2.0
mean2 = [1, 2, 1]
V2 = [[D2, D2 * r2, D2*r2*r2], [D2 * r2, D2, D2*r2], [D2*r2*r2, D2*r2, D2]]
# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# вычисление оценок математических ожиданий
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
# вычисление ковариационных матриц
VV1 = np.cov(x1)
VV2 = np.cov(x2)
# параметры для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

b = lambda x, mm, vv, Py, L: np.log(L * Py) - ((x-mm) @ np.linalg.inv(vv) @ (x-mm).T).diagonal() / 2 - np.log(np.linalg.det(vv)) / 2
b1, b2 = b(x_train, mm1, VV1, Py1, L1), b(x_train, mm2, VV2, Py2, L2)
predict = 2 * np.argmax((b1, b2), axis=0) - 1
loss = lambda a, y: a != y
Q = sum(loss(predict, y_train))

'''
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.scatter3D(x_train[y_train==1][:,0], x_train[y_train==1][:,1], x_train[y_train==1][:,2], color='blue', alpha=0.3)
ax.scatter3D(x_train[y_train==-1][:,0], x_train[y_train==-1][:,1], x_train[y_train==-1][:,2], color='green', alpha=0.3)
ax.scatter3D(x_train[y_train!=predict][:,0], x_train[y_train!=predict][:,1], x_train[y_train!=predict][:,2], color='red', alpha=0.8)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
fig, axs = plt.subplots(figsize=(15, 5), ncols=3)
axs[0].scatter(x_train[y_train==1][:,0], x_train[y_train==1][:,1], color='blue')
axs[0].scatter(x_train[y_train==-1][:,0], x_train[y_train==-1][:,1], color='green')
axs[0].scatter(x_train[y_train!=predict][:,0], x_train[y_train!=predict][:,1], color='red')
axs[0].set_xlabel('x1')
axs[0].set_ylabel('x2')
axs[1].scatter(x_train[y_train==1][:,0], x_train[y_train==1][:,2], color='blue')
axs[1].scatter(x_train[y_train==-1][:,0], x_train[y_train==-1][:,2], color='green')
axs[1].scatter(x_train[y_train!=predict][:,0], x_train[y_train!=predict][:,2], color='red')
axs[1].set_xlabel('x1')
axs[1].set_ylabel('x3')
axs[2].scatter(x_train[y_train==1][:,1], x_train[y_train==1][:,2], color='blue')
axs[2].scatter(x_train[y_train==-1][:,1], x_train[y_train==-1][:,2], color='green')
axs[2].scatter(x_train[y_train!=predict][:,1], x_train[y_train!=predict][:,2], color='red')
axs[2].set_xlabel('x2')
axs[2].set_ylabel('x3')
plt.show()
'''
```

## 3. Повторение: три класса и разные ковариации

**Задача на повторение / аналог предыдущей.**

Повторение гауссовской классификации для трёх классов с разными ковариациями. Параметры классов: (r,D,mu)=(0.7,3,[1,-2]), (0.5,2,[-3,-1]), (0.3,1,[1,2]). Использовать оценённые статистики из заготовки, реализовать полный гауссовский байесовский классификатор. Сохранить predict в исходном порядке и количество ошибок Q. Без вывода.

### Исходные данные и заготовка

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений трех классов
r1 = 0.7
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

r3 = 0.3
D3 = 1.0
mean3 = [1, 2]
V3 = [[D3, D3 * r3], [D3 * r3, D3]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
x3 = np.random.multivariate_normal(mean3, V3, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x3.T - mm3).T
VV3 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# параметры для гауссовского байесовского классификатора
Py1, Py2, Py3 = 0.2, 0.5, 0.3
L1, L2, L3 = 1, 1, 1
```

### Решение

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений трех классов
r1 = 0.7
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

r3 = 0.3
D3 = 1.0
mean3 = [1, 2]
V3 = [[D3, D3 * r3], [D3 * r3, D3]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
x3 = np.random.multivariate_normal(mean3, V3, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x3.T - mm3).T
VV3 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# параметры для гауссовского байесовского классификатора
Py1, Py2, Py3 = 0.2, 0.5, 0.3
L1, L2, L3 = 1, 1, 1

# здесь продолжайте программу
VV1_inv = np.linalg.inv(VV1)
VV2_inv = np.linalg.inv(VV2)
VV3_inv = np.linalg.inv(VV3)

model = lambda l, py, x, m, v, v_inv: l * py / np.sqrt(2 * np.pi * np.linalg.det(v)) * np.e**(-0.5 * (x - m).T @ v_inv @ (x - m))
am = lambda x: np.argmax([model(L1, Py1, x, mm1, VV1, VV1_inv),
                          model(L2, Py2, x, mm2, VV2, VV2_inv),
                          model(L3, Py3, x, mm3, VV3, VV3_inv)])

predict = [am(x) for x in x_train]

Q = sum(predict != y_train)
```
