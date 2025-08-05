- использование linalg для вычисления собственных чисел и собственных векторов

```python
import numpy as np

np.random.seed(0)

# исходные параметры для формирования образов обучающей выборки
r = 0.7
D = 3.0
mean = [3, 7, -2, 4, 6]
n_feature = 5
V = [[D * r ** abs(i - j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N = 1000
X = np.random.multivariate_normal(mean, V, N)

# здесь продолжайте программу
# X матрицу Грама для признаков (результирующий размер n_feature x n_feature)
F = 1 / N * X.T @ X
# вычислите собственные числа L и собственные векторы W
L, W = np.linalg.eig(F)
# сортировка собственных векторов
W = sorted(zip(L, W.T), key=lambda lx: lx[0], reverse=True)
W = np.array([w[1] for w in W])
# вычисление X в пространстве векторов WW.
G = X @ W.T
```

```python
import numpy as np


def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 3


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)
K = 10
X = np.array([[xx ** i for i in range(K)] for xx in coord_x])  # обучающая выборка для поиска коэффициентов модели
Y = coord_y

X_train = X[::2]  # обучающая выборка (входы)
Y_train = Y[::2]  # обучающая выборка (целевые значения)

# здесь продолжайте программу
N = len(X_train)
F = 1 / N * X_train.T @ X_train
L, W = np.linalg.eig(F)

WW = sorted(zip(L, W), key=lambda lx: lx[0], reverse=False)
WW = np.array([w[1] for w in WW])

# вычислите новые признаки G матрицы X в пространстве собственных векторов матрицы WW
G = X @ WW.T
# Оставьте в матрице G только первые 7 признаков
G = G[:, :7]
# Сформируйте матрицу XX_train из образов с новыми признаками G командой:
XX_train = G[::2]
# На основе выборки XX_train, Y_train вычислите вектор параметров w
w = np.linalg.inv(XX_train.T @ XX_train) @ XX_train.T @ Y_train
# Выполните восстановление функции func используя матрицу G и вектор параметров w
predict = G @ w
```

- pca с сортировкой

```python
import numpy as np

np.random.seed(0)

n_total = 1000  # число образов выборки
n_features = 200  # число признаков

table = np.zeros(shape=(n_total, n_features))

for _ in range(100):
    i, j = np.random.randint(0, n_total), np.random.randint(0, n_features)
    table[i, j] = np.random.randint(1, 10)

# матрицу table не менять

# здесь продолжайте программу
N = len(table)
F = 1 / N * table.T @ table
L, W = np.linalg.eig(F)

WW = sorted(zip(L, W), key=lambda lx: lx[0], reverse=False)
WW = np.array([w[1] for w in WW])

# в новом базисе
data_x = table @ WW.T
# фильтрация по массиву L
data_x = data_x[:, L >= 0.01]
```