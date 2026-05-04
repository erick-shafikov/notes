# Линейный Дискриминант Фишера

Вывод из гауссовского байесовского классификатора при допущении, что ковариационные матрицы **одинаковы для всех классов**:

$$\Sigma = \Sigma_{y_1} = \Sigma_{y_2} = \cdots = \Sigma_{y_K}$$

Оценки параметров:

$$\hat{\mu}_y = \frac{1}{l_y}\sum_{i:\,y_i = y} x_i, \quad y \in \mathcal{Y}$$

$$\hat{\Sigma} = \frac{1}{l}\sum_{y \in \mathcal{Y}}\sum_{i:\,y_i=y}(x_i - \hat{\mu}_y)(x_i - \hat{\mu}_y)^T \qquad \text{(по всей выборке)}$$

Общий гауссовский байесовский классификатор принимает вид:

$$a(x) = \arg\max_{y}\!\left(\ln\lambda_y P(y) - \frac{1}{2}(x - \hat{\mu}_y)^T\Sigma^{-1}(x - \hat{\mu}_y) - \frac{1}{2}\ln\det\Sigma_y\right)$$

Поскольку $\Sigma$ одинакова для всех классов, слагаемое $\frac{1}{2}\ln\det\Sigma$ не зависит от $y$ и выпадает из $\arg\max$. Раскрываем квадратичную форму:

$$-\frac{1}{2}(x - \mu_y)^T\Sigma^{-1}(x - \mu_y) = -\frac{1}{2}x^T\Sigma^{-1}x + x^T\Sigma^{-1}\mu_y - \frac{1}{2}\mu_y^T\Sigma^{-1}\mu_y$$

Слагаемое $-\frac{1}{2}x^T\Sigma^{-1}x$ тоже одинаково для всех $y$ и выпадает. Итого:

$$a(x) = \arg\max_{y}\!\left(\ln\lambda_y P(y) - \frac{1}{2}\mu_y^T\Sigma^{-1}\mu_y + x^T\Sigma^{-1}\mu_y\right)$$

Вводим обозначения:

$$\alpha_y = \Sigma^{-1}\mu_y, \qquad \beta_y = \ln\lambda_y P(y) - \frac{1}{2}\mu_y^T\Sigma^{-1}\mu_y$$

Тогда классификатор становится **линейным** по $x$:

$$a(x) = \arg\max_{y}\!\left(x^T\alpha_y + \beta_y\right)$$

При одинаковых ковариационных матрицах граница между классами — гиперплоскость. Это хорошо работает для несбалансированных классов, т.к. разница в $P(y)$ учитывается через $\beta_y$.

**Численная устойчивость.** Если $\hat{\Sigma}$ плохо обусловлена (близка к вырожденной), добавляют регуляризацию:

$$\hat{\Sigma} \leftarrow \hat{\Sigma} + \varepsilon I$$

— матрица становится гарантированно обратимой при любом $\varepsilon > 0$.

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