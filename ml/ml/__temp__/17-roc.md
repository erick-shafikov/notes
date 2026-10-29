# ROC-кривая и ранжирование

## 1. FPR и TPR при фиксированном пороге

**Задача на повторение / аналог предыдущей.**

Обучить SVC(kernel="linear") на заданном разбиении, сохранить w=[intercept_[0],*coef_[0]]. Для тестовых объектов при пороге t=2 вычислить FPR=FP/(FP+TN) и TPR=TP/(TP+FN). Записать в FPR и TPR; ничего не печатать.

### Исходные данные и заготовка

```python
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.2
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 1000
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.5, shuffle=True)
```

### Решение

```python
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.2
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 1000
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.5, shuffle=True)

t = 2
x_test = np.array([[1, x[0], x[1]] for x in x_test])
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
w = np.hstack((clf.intercept_[0], clf.coef_[0]))
predict = np.sign(w @ x_test.T - t)

TP = predict[(predict == 1) & (y_test == 1)].shape[0]
TN = predict[(predict == -1) & (y_test == -1)].shape[0]
FP = predict[(predict == 1) & (y_test == -1)].shape[0]
FN = predict[(predict == -1) & (y_test == 1)].shape[0]

FPR = FP / (FP + TN)
TPR = TP / (TP + FN)
```

## 2. ROC-кривая по сетке порогов

**Задача на повторение / аналог предыдущей.**

Обучить линейный SVC на исходной выборке. Для каждого t из np.arange(5.7,-7.8,-0.1) вычислить FPR(t) и TPR(t) на тесте. Координаты ROC-кривой сохранить в списках FPR и TPR в порядке порогов; вывод не нужен.

### Исходные данные и заготовка

```python
range_t = np.arange(5.7, -7.8, -0.1)
```

### Решение

```python
from multiprocessing import reduction
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

np.random.seed(0)

# Исходные параметры распределений классов
r1 = -0.2
D1 = 3.0
mean1 = [1, -5]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -2]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# Моделирование обучающей выборки
N1 = 1000
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.5, shuffle=True)

# Обучение модели SVM с линейным ядром
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

# Получение коэффициентов весов
w = np.hstack([clf.intercept_[0], clf.coef_[0]])

# Вычисление значений ROC-кривой
range_t = np.arange(5.7, -7.8, -0.1)
FPR = []
TPR = []

# Получение вероятностей принадлежности к положительному классу
y_scores = clf.decision_function(x_test)

# Вычисление ROC-кривой
for t in range_t:
    y_pred = np.where(y_scores > t, 1, -1)
    
    # Вычисление матрицы ошибок
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    FPR.append(fp / (fp + tn))
    TPR.append(tp / (tp + fn))
```
