# Линейный дискриминант Фишера

## 1. Линейный дискриминант двух классов

Для двумерных гауссовских классов с общей корреляцией 0.7 и дисперсией 2, средними [1,-2] и [1,3], реализовать линейный дискриминант Фишера. Использовать оценённые по выборке средние и общую ковариацию. Штрафы равны 1, априорные вероятности 0.5. Сохранить метки обучающих объектов в predict и число ошибок в Q. Вывод не нужен.

### Решение

```python
import numpy as np

np.random.seed(0)

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
VV = np.array([[np.dot(a[0], a[0]) / (2*N), np.dot(a[0], a[1]) / (2*(N - 1))],
                [np.dot(a[1], a[0]) / (2*(N - 1)), np.dot(a[1], a[1]) / (2*(N - 1))]])

cov_inv = np.linalg.inv(VV)


# Функция для классификации с использованием линейного дискриминанта Фишера
def fisher_discriminant(x):
    p1 = - 0.5 * (mm1.T @ cov_inv @ mm1) + (x.T @ cov_inv @ mm1)
    p2 = - 0.5 * (mm2.T @ cov_inv @ mm2) + (x.T @ cov_inv @ mm2)
    return np.unique(y_train)[np.argmax([p1, p2])]

# Классификация обучающей выборки
predict = [fisher_discriminant(x) for x in x_train]

# Вычисление показателя качества
Q = sum(predict != y_train)
```

## 2. Коэффициенты линейных дискриминантов

По заданной выборке двух классов 0 и 1 оценить средние mu_y и общую ковариационную матрицу Sigma. Найти alpha_y=Sigma^(-1)*mu_y и beta_y=log(lambda_y*P_y)-0.5*mu_y^T*Sigma^(-1)*mu_y. Сохранить alpha1, alpha2 (векторы), beta1, beta2 (скаляры) в порядке классов 0,1. Настройки априорных вероятностей и штрафов — в заготовке. Не печатать.

### Исходные данные и заготовка

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2, 0])
mean2 = np.array([1, 3, 1])
r = 0.7
D = 2.0
V = [[D, D * r, D*r*r], [D*r, D, D*r], [D*r*r, D*r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.zeros(N), np.ones(N)])

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2

# параметры для линейного дискриминанта Фишера
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации
```

### Решение

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2, 0])
mean2 = np.array([1, 3, 1])
r = 0.7
D = 2.0
V = [[D, D * r, D*r*r], [D*r, D, D*r], [D*r*r, D*r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.zeros(N), np.ones(N)])

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2

# параметры для линейного дискриминанта Фишера
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

# здесь продолжайте программу
# вычисление оценок математических ожиданий
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

# вычисление ковариационных матриц
a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N, np.dot(a[0], a[2]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N, np.dot(a[1], a[2]) / N],
                [np.dot(a[2], a[0]) / N, np.dot(a[2], a[1]) / N, np.dot(a[2], a[2]) / N]])
a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N, np.dot(a[0], a[2]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N, np.dot(a[1], a[2]) / N],
                [np.dot(a[2], a[0]) / N, np.dot(a[2], a[1]) / N, np.dot(a[2], a[2]) / N]])
VV = (VV1 + VV2) / 2

alpha1 = np.linalg.inv(VV) @ mm1
alpha2 = np.linalg.inv(VV) @ mm2

beta1 = np.log(L1 * Py1) - 0.5 * mm1.T @ alpha1
beta2 = np.log(L2 * Py2) - 0.5 * mm2.T @ alpha2
```

## 3. Три класса с общей ковариацией

Реализовать дискриминант Фишера для трёх двумерных классов {0,1,2}: общие r=0.5, D=1, средние [1,-2],[-3,-1],[1,2]. По статистикам из заготовки вычислить линейные дискриминанты x^T*alpha_y+beta_y, выбрать максимальный. Сохранить predict для всех обучающих объектов и количество ошибок Q; не печатать.

### Решение

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2])
mean2 = np.array([-3, -1])
mean3 = np.array([1, 2])

r = 0.5
D = 1.0
V = [[D, D * r], [D*r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T
x3 = np.random.multivariate_normal(mean3, V, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

mm = [np.mean(xx, axis=0) for xx in [x1.T, x2.T, x3.T]]
A = np.vstack([(x1.T - mm[0]), (x2.T - mm[1]), (x3.T - mm[2])])
VV = np.cov(A.T)
VV_inv = np.linalg.inv(VV)

# параметры для линейного дискриминанта Фишера
Py = [0.2, 0.4, 0.4]
lm = [1, 1, 1]

alpha = [VV_inv @ mm[i] for i in range(3)]
beta = [np.log(lm[i] * Py[i]) - 0.5 * (mm[i].T @ VV_inv @ mm[i]) for i in range(3)]

def a(x):
    p_012 = [x.T @ alpha[i] + beta[i] for i in range(3)]
    return np.argmax(p_012)

predict = [a(xx) for xx in x_train]
Q = np.sum(predict != y_train)
```
