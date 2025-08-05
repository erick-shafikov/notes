- использование линейного дискриминанта фишера для двух классификаций

```python
import numpy as np

# исходные параметры распределений двух классов
mean1 = [1, -2]
mean2 = [1, 3]
r = 0.7
D = 2.0
V = [[D, D * r], [D * r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# вычисление оценок МО и ковариационной матрицы
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
VV = np.array([[np.dot(a[0], a[0]) / (2 * N), np.dot(a[0], a[1]) / (2 * N)],
               [np.dot(a[1], a[0]) / (2 * N), np.dot(a[1], a[1]) / (2 * N)]])

# здесь продолжайте программу
inv_VV = np.linalg.inv(VV)

ax = lambda x, mm: np.log(1 * 0.5) - 0.5 * mm.T @ inv_VV @ mm + x.T @ inv_VV @ mm

predict = []

for x in x_train:
    predict.append(np.argmax([ax(x, mm1), ax(x, mm2)]) * 2 - 1)

Q = np.sum(predict != y_train)
predict = np.array(predict)

```

- с тремя классификациями

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2])
mean2 = np.array([-3, -1])
mean3 = np.array([1, 2])

r = 0.5
D = 1.0
V = [[D, D * r], [D * r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T
x3 = np.random.multivariate_normal(mean3, V, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2, x3

# параметры для линейного дискриминанта Фишера
Py1, Py2, Py3 = 0.2, 0.4, 0.4
L1, L2, L3 = 1, 1, 1

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T, (x3.T - mm3).T])

VV = np.array([[np.dot(a[0], a[0]) / (3 * N), np.dot(a[0], a[1]) / (3 * N)],
               [np.dot(a[1], a[0]) / (3 * N), np.dot(a[1], a[1]) / (3 * N)]])

inv_VV = np.linalg.inv(VV)

alpha = lambda mm: inv_VV @ mm
beta = lambda l, p, mm: np.log(l * p) - 0.5 * mm.T @ inv_VV @ mm
model = lambda x, mm, l, p: x.T @ alpha(mm) + beta(l, p, mm)

predict = []

for x in x_train:
    predict.append(np.argmax([model(x, mm1, L1, Py1), model(x, mm2, L2, Py2), model(x, mm3, L3, Py3)]))

Q = np.sum(predict != y_train)

```

- вычисление параметров для модели Фишера

```python

import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2, 0])
mean2 = np.array([1, 3, 1])
r = 0.7
D = 2.0
V = [[D, D * r, D * r * r], [D * r, D, D * r], [D * r * r, D * r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.zeros(N), np.ones(N)])

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
VV = np.array(
    [[np.dot(a[0], a[0]) / (N * 2 - 1), np.dot(a[0], a[1]) / (N * 2 - 1), np.dot(a[0], a[2]) / (N * 2 - 1)],
     [np.dot(a[1], a[0]) / (N * 2 - 1), np.dot(a[1], a[1]) / (N * 2 - 1), np.dot(a[1], a[2]) / (N * 2 - 1)],
     [np.dot(a[2], a[0]) / (N * 2 - 1), np.dot(a[2], a[1]) / (N * 2 - 1), np.dot(a[2], a[2]) / (N * 2 - 1)]],
)

inv_VV = np.linalg.inv(VV)
# параметры для линейного дискриминанта Фишера
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

alpha1 = inv_VV @ mm1
alpha2 = inv_VV @ mm2
beta1 = np.log(Py1 * L1) - 0.5 * mm1.T @ inv_VV @ mm1
beta2 = np.log(Py2 * L2) - 0.5 * mm2.T @ inv_VV @ mm2
```