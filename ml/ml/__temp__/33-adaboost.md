# AdaBoost для классификации

## 1. Ручная композиция AdaBoost

Реализовать AdaBoost из T=10 DecisionTreeClassifier(max_depth=3, criterion="gini") на заданном train/test. Начальные веса объектов равны 1/l. На каждом шаге обучать с sample_weight, найти взвешенную ошибку N, alpha=0.5*log((1-N)/N), умножить веса на exp(-alpha*y*b(x)) и нормировать. На тесте предсказывать sign(sum(alpha_t*b_t(x))). Сохранить predict и число ошибок Q; не печатать.

### Исходные данные и заготовка

```python
b_t = DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
```

### Решение

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


np.random.seed(0)
n_feature = 2

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 7]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [4, 2]
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N1, N2 = 1000, 1200
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

max_depth = 3
b_t = DecisionTreeClassifier(criterion='gini', max_depth=max_depth)

w = np.ones(len(x_train)) / len(x_train)
algs = []               # список из полученных алгоритмов
alfa = []               # список из вычисленных весов для композиции
T = 10

for t in range(T):
    algs.append(DecisionTreeClassifier(criterion='gini', max_depth=max_depth))
    algs[t].fit(x_train, y_train, sample_weight=w)
    predicted = algs[t].predict(x_train)
    N = np.sum((y_train != predicted) * w) + 1e-8
    alfa.append(0.5 * np.log((1 - N) / N))
    w *= np.exp(-alfa[t] * y_train * predicted)
    w /= w.sum()
    

predict = alfa[0] * algs[0].predict(x_test)
for n in range(1, T):
    predict += alfa[n] * algs[n].predict(x_test)

predict = np.sign(predict)
Q = np.sum(predict != y_test)
```
