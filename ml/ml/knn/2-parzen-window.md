```python
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
n_feature = 2  # число признаков

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 3]
V1 = [[D1 * r1 ** abs(i - j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [1, 1]  # + np.array(range(1, n_feature+1)) * 0.5
V2 = [[D2 * r2 ** abs(i - j) for j in range(n_feature)] for i in range(n_feature)]

r3 = -0.7
D3 = 1.0
mean3 = [-2, -2]  # + np.array(range(1, n_feature+1)) * -0.5
V3 = [[D3 * r3 ** abs(i - j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N1, N2, N3 = 200, 150, 190
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
x3 = np.random.multivariate_normal(mean3, V3, N3).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N1), np.ones(N2), np.ones(N3) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.5, shuffle=True)

h = 1


def get_dist(x1, x2):
    return np.sum(np.abs(x1 - x2))


def Kr(r):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-1 * np.square(r) / 2)


def ax(yi, xi):
    return np.sum([Kr(get_dist(x_test_i, xi) / h) * (y_train[i] == yi) for i, x_test_i in enumerate(x_train)])


predict = [int(np.argmax([ax(0, x), ax(1, x), ax(2, x)])) for x in x_test]

Q = (predict != y_test).mean()

```