# Гауссовский Байесовский классификатор

## Многомерный случай

Выборка $X^l = \{(x_i, y_i)\}_{i=1}^l$, где $x \in \mathbb{R}^n$. Предполагается, что объекты каждого класса порождены **многомерным нормальным распределением**:

$$P(x \mid y) = \frac{1}{(2\pi)^{n/2}(\det\Sigma_y)^{1/2}} \exp\!\left\{-\frac{1}{2}(x - \mu_y)^T \Sigma_y^{-1}(x - \mu_y)\right\}$$

где:

- $\mu_y = [\mu_y^1,\, \mu_y^2,\, \ldots,\, \mu_y^n]^T$ — вектор математических ожиданий признаков для класса $y$
- $\Sigma_y = \mathbb{E}\!\left[(x - \mu_y)(x - \mu_y)^T\right]$ — ковариационная матрица класса $y$

Классификатор:

$$a(x) = \arg\max_{y} \lambda_y\, P(y) \cdot p(x \mid y)$$

По структуре это тот же Наивный Байес, но вместо скалярных дисперсий используется полная ковариационная матрица $\Sigma_y$, которая учитывает зависимости между признаками. Матрица $\Sigma_y$ должна быть **невырожденной (обратимой)**.

Переходя к логарифму (логарифм от гауссианы):

$$a(x) = \arg\max_{y} \left(\ln\lambda_y P(y) - \frac{1}{2}(x - \hat{\mu}_y)^T \Sigma_y^{-1}(x - \hat{\mu}_y) - \frac{1}{2}\ln\det\Sigma_y\right)$$

## Связь с Наивным Байесом: диагональная ковариация

Если признаки независимы, ковариационная матрица диагональна:

$$\Sigma = \begin{pmatrix}\sigma_{x_1}^2 & 0 & \cdots & 0 \\ 0 & \sigma_{x_2}^2 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \cdots & 0 & \sigma_{x_n}^2\end{pmatrix}$$

Тогда квадратичная форма распадается в сумму:

$$P(x \mid y) = \frac{1}{(2\pi\det\Sigma_y)^{n/2}} \exp\!\left\{-\sum_{i=1}^n \frac{(x_i - m_{y,i})^2}{2\sigma_{x_i}^2}\right\} = \prod_{i=1}^n p(x_i \mid y)$$

что в точности совпадает с Наивным Байесом. Т.е. Наивный Байес — частный случай гауссовского байесовского классификатора с диагональной $\Sigma_y$.

## Ковариация и корреляция

$$B(x, y) = \mathbb{E}\{(x - m_x)(y - m_y)\} = \iint (x - m_x)(y - m_y)\, p(x, y)\, dx\, dy$$

$$R(x, y) = \frac{B(x, y)}{\sigma_x\,\sigma_y}, \qquad -1 \leq R(x, y) \leq 1$$

Если $R = 0$ — признаки некоррелированы (при гауссовом распределении — независимы), ковариационная матрица диагональна.

## Оценка параметров по выборке (MLE)

$$\hat{\mu}_y = \frac{1}{l_y}\sum_{i:\,y_i=y} x_i \qquad \text{(выборочное среднее)}$$

$$\hat{\Sigma}_y = \frac{1}{l_y}\sum_{i:\,y_i=y}(x_i - \hat{\mu}_y)(x_i - \hat{\mu}_y)^T \qquad \text{(выборочная ковариационная матрица)}$$

где $l_y$ — число объектов класса $y$.

## Пример (2 класса, 2 признака)

Обучение: 100 объектов на класс. Параметры:

|            | $r$   | $\sigma^2$ | $\mu_y$      |
| ---------- | ----- | ---------- | ------------ |
| класс $-1$ | $0.7$ | $1.0$      | $[0,\,-3]^T$ |
| класс $+1$ | $0.7$ | $2.0$      | $[0,\,+3]^T$ |

Ковариационные матрицы:

$$\Sigma_{y_1} = \begin{pmatrix}\sigma_{x1}^2 & \sigma_{x1}^2\, r_1 \\ \sigma_{x1}^2\, r_1 & \sigma_{x1}^2\end{pmatrix}, \qquad \Sigma_{y_2} = \begin{pmatrix}\sigma_{x2}^2 & \sigma_{x2}^2\, r_2 \\ \sigma_{x2}^2\, r_2 & \sigma_{x2}^2\end{pmatrix}$$

## Геометрическая интерпретация: эллипс рассеивания

При гауссовом распределении линии равной плотности — **эллипсы**. Направления осей задаёт собственное разложение ковариационной матрицы:

$$\Sigma = V \cdot S \cdot V^T$$

где $V$ — матрица собственных векторов (направления осей эллипса), $S$ — диагональная матрица собственных значений (разброс вдоль каждой оси).

Применяя ковариационную матрицу $\Sigma$, мы переходим в **новую систему координат**, в которой признаки некоррелированы. В этих координатах граница классов определяется оптимальным байесовским решающим правилом.

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

# здесь продолжайте программу
ax = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

length = len(x_train)
predict = []
for x in x_train:
    predict.append(np.argmax([ax(x, VV1, mm1, L1, Py1), ax(x, VV2, mm2, L2, Py2)]) * 2 - 1)

Q = np.sum(predict != y_train)
predict = np.array(predict)

```

3 X 3

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [-1, -2, -1]
V1 = [[D1, D1 * r1, D1 * r1 * r1], [D1 * r1, D1, D1 * r1], [D1 * r1 * r1, D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [1, 2, 1]
V2 = [[D2, D2 * r2, D2 * r2 * r2], [D2 * r2, D2, D2 * r2], [D2 * r2 * r2, D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# здесь продолжайте программу
# вычисление оценок математических ожиданий
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

# вычисление ковариационных матриц
a = (x1.T - mm1).T
VV1 = np.array(
    [[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N, np.dot(a[0], a[2]) / N],
     [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N, np.dot(a[1], a[2]) / N],
     [np.dot(a[2], a[0]) / N, np.dot(a[2], a[1]) / N, np.dot(a[2], a[2]) / N]],
)

a = (x2.T - mm2).T
VV2 = np.array(
    [[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N, np.dot(a[0], a[2]) / N],
     [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N, np.dot(a[1], a[2]) / N],
     [np.dot(a[2], a[0]) / N, np.dot(a[2], a[1]) / N, np.dot(a[2], a[2]) / N]]
)

Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

ax = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

predict = []
for x in x_train:
    predict.append(np.argmax([ax(x, VV1, mm1, L1, Py1), ax(x, VV2, mm2, L2, Py2)]) * 2 - 1)

Q = np.sum(predict != y_train)
```

- для трех классификаций

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
ax = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

predict = []

for x in x_train:
    predict.append(np.argmax([ax(x, VV1, mm1, L1, Py1), ax(x, VV2, mm2, L2, Py2), ax(x, VV3, mm3, L3, Py3)]))

Q = np.sum(predict != y_train)

```
