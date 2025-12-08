# Перемножение матриц

```python
import torch

a = torch.arange(1, 2).view(3, 3)
b = torch.arange(10, 19).view(3, 3)
# поэлементное умножение
r1 = a * b
r2 = torch.mul(a, b)
# матричное умножение
c_mat_mul = torch.matmul(a, b)  # с возможностью транслирования
c_at = a @ b
c_mm = torch.mm(a, b)  # без возможности транслирования
# транслирования
# matmul дает возможность перемножать вектор на матрицу
v = torch.tensor([-1, 2, 3])
torch.matmul(a, v)
torch.matmul(v, a)
# mm пробросит ошибку
error_mm = torch.mm(v, a)
# альтернативные варианты записи
c_mm_1 = a.mm(b)
c_mat_mul_1 = a.matmul(b)

# bmm - для перемножения пакета матриц, batch
bx = torch.randn(7, 3, 5)  # 7 матриц 3 * 5
by = torch.randn(7, 5, 4)  # 7 матриц 5 * 4
bc = torch.bmm(bx, by)  # 7 матриц 3 * 4 т.к 3х5 * 5х4 = 3*4 перемножаться 1 и 1 матрицы, 2 и 2 итд

# Умножение векторов
a = torch.arange(1, 10)
b = torch.ones(9)
# Скалярное произведение
#                     [b1,
# [a1, a2, ..., an] *  b2 = a1 * b1 + a2 * b2 + ... an * bn
#                     ...
#                      bn]

c_dot = torch.dot(a, b)

# внешнее произведение
# [a1,                     [a1 * b1, a1 * b2, ... a1 * bn
#  a2, * [b1, b2, ...bn] =  a2 * b1, a2 * b2  ...
# ...                                 ...
#  an]                      an * b1, an * b2  ... an * bn ]

c_outer = torch.outer(a, b)

# Умножение вектора на матрицу
matrix = torch.tensor([[1, 2], [2, 3]])
vector = torch.tensor([1, 2])
# вариант 1 - вектор * матрицу => вектор * столбец = вектор
# вариант 2 - матрица * вектор => строки матрицы * вектор = вектор
mv = torch.mv(matrix, vector)
mv_1 = matrix.mv(vector)
```

# Вычисление матричных характеристик

linalg.svd() - сингулярное (SVD) разложение матрицы
linalg.det() - определитель (детерминант) матрицы
torch.trace() - сумма элементов главной диагонали матрицы
linalg.eig() - вычисление собственных значений и правых собственных векторов
linalg.eigvals() - вычисление собственных значений матрицы
linalg.inv() - вычисление обратной матрицы

# Системы линейных уравнений

```python
import torch

# матрица - коэффициенты
eq_coef = torch.FloatTensor([(1, 2, 3), (1, 4, 9), (1, 8, 27)])
y = torch.FloatTensor([10, 20, 30])
# ранг
rank = torch.linalg.matrix_rank(eq_coef)  # 3 - независимы
# решение
solve_x = torch.linalg.solve(eq_coef, y)
# решение через обратную
inv_coef = torch.linalg.inv(eq_coef)
solve_x_inv = torch.mv(inv_coef, y)

```

# Нейронная сеть

Двухслойная нейронная сеть с сигмоидной функцией активации для скрытого слоя и линейной функцией активации у выходного
нейрона:

```python
import torch

# значения списков w и g в программе не менять
w = list(map(float, input().split()))
g = list(map(float, input().split()))

W = torch.tensor(w).resize_(2, -1)
W1 = W[:, 1:]
bias1 = W[:, 0]

G = torch.tensor(g)
W2 = G[1:]
bias2 = G[0]

t_inp = torch.rand(3) * 10

u = torch.sigmoid(W1 @ t_inp + bias1)
y = W2 @ u + bias2
```