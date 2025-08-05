```python
import numpy as np
from sklearn import svm


def func(x):
    return np.sin(0.5 * x) + 0.2 * np.cos(2 * x) - 0.1 * np.sin(4 * x) + 3


# обучающая выборка
coord_x = np.expand_dims(np.arange(-4.0, 6.0, 0.1), axis=1)
coord_y = func(coord_x).ravel()

# здесь продолжайте программу
x_train = coord_x[::3]
y_train = coord_y[::3]

svr = svm.SVR(kernel='rbf')
svr.fit(x_train, y_train)
# обучение
predict = svr.predict(coord_x)
Q = np.square(predict - coord_y).mean()
```

```python
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.6
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-2, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 500
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.4, shuffle=True)

# здесь продолжайте программу
clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)
predict = clf.predict(x_test)

Q = (predict != y_test).mean()
acc = (predict == y_train).mean()  # показатель аккуратности модели
```