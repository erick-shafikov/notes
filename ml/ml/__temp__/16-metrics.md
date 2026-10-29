# Оценка качества классификации

## 1. Accuracy линейного SVM

Обучить линейный SVC на x_train,y_train с исходными настройками. Сохранить коэффициенты гиперплоскости [w0,w1,w2] в w, а долю правильных ответов на x_test,y_test — в acc. Не печатать.

### Исходные данные и заготовка

```python
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.3
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 5.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.4, shuffle=True)
```

### Решение

```python
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = 0.3
D1 = 3.0
mean1 = [2, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 5.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.4, shuffle=True)

# здесь продолжайте программу
clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train)

w = [clf.intercept_[0], *clf.coef_[0]]

predict = np.array(clf.predict(x_test))

acc = np.mean(y_test == predict)
```

## 2. Элементы матрицы ошибок

В тесте 25 объектов класса +1 и 75 класса -1. Из положительных 20 распознаны как +1, 5 как -1; из отрицательных 60 распознаны как -1, 15 как +1. Определить TP,TN,FP,FN и записать в одноимённые переменные; не печатать.

### Решение

```python
# запишите вычисленные значения
TP = 20
TN = 60
FP = 15
FN = 5
```

## 3. Гауссовский классификатор и матрица ошибок

**Задача на повторение / аналог предыдущей.**

Повторить гауссовскую байесовскую классификацию заданных двумерных данных с полными ковариациями, используя log(lambda_y*P_y)-0.5*(x-mu_y)^T*Sigma_y^(-1)*(x-mu_y)-0.5*log(det(Sigma_y)). Сохранить predict в исходном порядке. Рассчитать TP,TN,FP,FN для класса +1 и сохранить в одноимённых переменных, без вывода.

### Исходные данные и заготовка

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = -0.5
D2 = 2.0
mean2 = [0, 2]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 500
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

# вычисление оценок МО и ковариационных матриц
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N1, np.dot(a[0], a[1]) / N1],
                [np.dot(a[1], a[0]) / N1, np.dot(a[1], a[1]) / N1]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N2, np.dot(a[0], a[1]) / N2],
                [np.dot(a[1], a[0]) / N2, np.dot(a[1], a[1]) / N2]])

# для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации
```

### Решение

```python
import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = -0.5
D2 = 2.0
mean2 = [0, 2]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 500
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

# вычисление оценок МО и ковариационных матриц
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N1, np.dot(a[0], a[1]) / N1],
                [np.dot(a[1], a[0]) / N1, np.dot(a[1], a[1]) / N1]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N2, np.dot(a[0], a[1]) / N2],
                [np.dot(a[1], a[0]) / N2, np.dot(a[1], a[1]) / N2]])

# для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1     # вероятности появления классов
Py2, L2 = 1 - Py1, 1 # и величины штрафов неверной классификации

# Функции:
# вычисление псевдовероятности
b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))
# argmax от вероятностей
am = lambda x: np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2)]) * 2 - 1 # классификатор

# Предсказание по модели
predict = [am(x) for x in data_x]

TP = sum([p == y and p == 1 for p, y in zip(predict, data_y)])
TN = sum([p == y and p == -1 for p, y in zip(predict, data_y)])
FP = sum([p != y and p == 1 for p, y in zip(predict, data_y)])
FN = sum([p != y and p == -1 for p, y in zip(predict, data_y)])

'''ГРАФИК'''
# Визуализация данных и разделяющей поверхности
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Создаем сетку для построения разделяющей поверхности
x_min, x_max = data_x[:, 0].min() - 1, data_x[:, 0].max() + 1
y_min, y_max = data_x[:, 1].min() - 1, data_x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Предсказание для каждой точки на сетке
Z = np.array([am(np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)

# Цветовая карта для классов
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# Построение контура разделяющей поверхности
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

# Цветовая карта для классов
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# Построение scatter plot для исходных данных с цветами классов
plt.scatter(data_x[:, 0], data_x[:, 1], c=data_y, cmap=cmap_bold, edgecolor='k', s=20)

# Выделение неправильно классифицированных точек
incorrect = (predict != data_y)
plt.scatter(data_x[incorrect, 0], data_x[incorrect, 1], color='yellow', edgecolor='k', s=100, label='Ошибки')

# Отображение центров классов
plt.scatter(mm1[0], mm1[1], color='red', marker='x', s=200, label='Центр класса 1')
plt.scatter(mm2[0], mm2[1], color='blue', marker='x', s=200, label='Центр класса -1')

# Добавление таблицы с результатами классификации
table_data = [["", "y = 1", "y = -1"],
              ["a(x) = 1", TP, FP],
              ["a(x) = -1", FN, TN]]

table = plt.table(cellText=table_data,
                  loc='upper right',
                  colWidths=[0.15] *3)


plt.title("Гауссовский байесовский классификатор")
plt.legend(loc='lower right')
plt.show()
```

## 4. Precision и recall после обучения SGD

**Задача на повторение / аналог предыдущей.**

Повторение SGD: по одному случайному объекту обучить линейный классификатор с потерей log2(1+exp(-M)). Использовать заданные данные и параметры. Сохранить веса в w, а precision и recall по обучающей выборке — в одноимённые переменные. Не печатать.

### Исходные данные и заготовка

```python
k = np.random.randint(0, n_train-1) # n_train - размер выборки (массива x_train)
```

### Решение

```python
import numpy as np

# логарифмическая функция потерь: log2​(1+e−{w^T⋅xi​⋅yi}​)
loss = lambda w, x, y: np.log2(1 + np.exp(x @ w * y))

# её производная вектору w: −e−{w^T⋅xi​⋅yi​⋅x^iT​⋅yi}​​ / ((1+e^{-wT⋅xi​⋅yi}​)⋅ln(2))
df = lambda w, x, y: -np.exp(-x @ w * y) * x * y / (1 + np.exp(-x @ w * y)) / np.log(2)

data_x = [(5.8, 1.2), (5.6, 1.5), (6.5, 1.5), (6.1, 1.3), (6.4, 1.3), (7.7, 2.0), (6.0, 1.8), (5.6, 1.3), (6.0, 1.6), (5.8, 1.9), (5.7, 2.0), (6.3, 1.5), (6.2, 1.8), (7.7, 2.3), (5.8, 1.2), (6.3, 1.8), (6.0, 1.0), (6.2, 1.3), (5.7, 1.3), (6.3, 1.9), (6.7, 2.5), (5.5, 1.2), (4.9, 1.0), (6.1, 1.4), (6.0, 1.6), (7.2, 2.5), (7.3, 1.8), (6.6, 1.4), (5.6, 2.0), (5.5, 1.0), (6.4, 2.2), (5.6, 1.3), (6.6, 1.3), (6.9, 2.1), (6.8, 2.1), (5.7, 1.3), (7.0, 1.4), (6.1, 1.4), (6.1, 1.8), (6.7, 1.7), (6.0, 1.5), (6.5, 1.8), (6.4, 1.5), (6.9, 1.5), (5.6, 1.3), (6.7, 1.4), (5.8, 1.9), (6.3, 1.3), (6.7, 2.1), (6.2, 2.3), (6.3, 2.4), (6.7, 1.8), (6.4, 2.3), (6.2, 1.5), (6.1, 1.4), (7.1, 2.1), (5.7, 1.0), (6.8, 1.4), (6.8, 2.3), (5.1, 1.1), (4.9, 1.7), (5.9, 1.8), (7.4, 1.9), (6.5, 2.0), (6.7, 1.5), (6.5, 2.0), (5.8, 1.0), (6.4, 2.1), (7.6, 2.1), (5.8, 2.4), (7.7, 2.2), (6.3, 1.5), (5.0, 1.0), (6.3, 1.6), (7.7, 2.3), (6.4, 1.9), (6.5, 2.2), (5.7, 1.2), (6.9, 2.3), (5.7, 1.3), (6.1, 1.2), (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), (7.9, 2.0), (5.6, 1.1), (7.2, 1.8), (5.5, 1.3), (7.2, 1.6), (6.3, 2.5), (6.3, 1.8), (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), (5.5, 1.3), (5.5, 1.1), (5.9, 1.5), (6.0, 1.5), (5.9, 1.8)]
data_y = [-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1]

x_train = np.array([[1, x[0], x[1]] for x in data_x])
y_train = np.array(data_y)

n_train = len(x_train)  # размер обучающей выборки
w = [0.0, 0.0, 0.0]  # начальные весовые коэффициенты
nt = np.array([0.5, 0.01, 0.01])  # шаг обучения для каждого параметра w0, w1, w2
lm = 0.01  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
N = 500  # число итераций алгоритма SGD

np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for _ in range(N):
    k = np.random.randint(0, n_train-1) # n_train - размер выборки (массива x_train)
    grad = df(w, x_train[k], y_train[k])
    w -= nt * grad

# Предсказание по модели
predict = np.sign(x_train @ w)

TP = sum([p == y and p == 1 for p, y in zip(predict, y_train)])
TN = sum([p == y and p == -1 for p, y in zip(predict, y_train)])
FP = sum([p != y and p == 1 for p, y in zip(predict, y_train)])
FN = sum([p != y and p == -1 for p, y in zip(predict, y_train)])

precision = TP / (TP + FP)
recall = TP / (TP + FN)
```

## 5. SVM, F1 и F-beta

**Задача на повторение / аналог предыдущей.**

Обучить линейный SVC с настройками заготовки, сохранить [intercept_[0],*coef_[0]] в w. На тестовой выборке вычислить TP,TN,FP,FN,precision,recall, F1 и F-beta при beta=0.5. Сохранить результаты в TP,TN,FP,FN,precision,recall,F,Fb. Ничего не печатать.

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
N1 = 2500
N2 = 1500
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.4, shuffle=True)
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
N1 = 2500
N2 = 1500
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.4, shuffle=True)

clf = svm.SVC(kernel='linear').fit(x_train, y_train) # объект clf + обучение

w = [clf.intercept_[0], *clf.coef_[0]]

# Предсказание по модели
predict = clf.predict(x_test)

TP = sum([p == y and p == 1 for p, y in zip(predict, y_test)])
TN = sum([p == y and p == -1 for p, y in zip(predict, y_test)])
FP = sum([p != y and p == 1 for p, y in zip(predict, y_test)])
FN = sum([p != y and p == -1 for p, y in zip(predict, y_test)])

precision = TP / (TP + FP)
recall = TP / (TP + FN)
# precision, recall
F = 2 * precision * recall / (precision + recall)
b = .5
Fb = (1 + b**2) * precision * recall / (b**2 * precision + recall)
```

## 6. Повторение: матрица ошибок для 160 объектов

**Задача на повторение / аналог предыдущей.**

Повторение матрицы ошибок: в тесте 110 положительных объектов (90 предсказаны как +1, 20 как -1) и 50 отрицательных (45 как -1, 5 как +1). Сохранить TP,TN,FP,FN в одноимённые переменные; не печатать.

### Решение

```python
import numpy as np

pos = np.array(np.ones(110))
neg = np.array([-1 for i in range(50)])

y = np.hstack((pos,neg))


predict = np.array(np.ones(90))
predict = np.hstack((predict,np.array(np.ones(20) * -1)))
predict = np.hstack((predict,np.array(np.ones(45) * -1)))
predict = np.hstack((predict,np.array(np.ones(5))))


TP = len([(a,b) for a,b in zip(predict,y) if a==b and a==1])
TN = len([(a,b) for a,b in zip(predict,y) if a==b and a==-1])

FP = len([(a,b) for a,b in zip(predict,y) if a!=b and a==1])
FN = len([(a,b) for a,b in zip(predict,y) if a!=b and a==-1])
```

## 7. Повторение: precision и recall для 220 объектов

**Задача на повторение / аналог предыдущей.**

Повторение precision и recall: из 70 объектов +1 правильно распознаны 65, ошибочно 5; из 150 объектов -1 правильно распознаны 120, ошибочно 30. Рассчитать precision и recall положительного класса и сохранить в одноимённых переменных. Без вывода.

### Решение

```python
TP = 65
TN = 120
FP = 30
FN = 5
precision = TP/(TP+FP)
recall = TP/(TP+FN)
```

## 8. Повторение: матрица ошибок для 161 объекта

**Задача на повторение / аналог предыдущей.**

Повторение: в тесте 87 положительных объектов (81 верный прогноз +1 и 6 ошибок) и 74 отрицательных (72 верных -1 и 2 ошибки). Определить TP,TN,FP,FN, записать в одноимённые переменные; не печатать.

### Решение

```python
TP = 81
TN = 72
FP = 2
FN = 6
```

## 9. Повторение: precision и recall для 300 объектов

**Задача на повторение / аналог предыдущей.**

Повторение: из 100 положительных объектов правильно распознаны 87, из 200 отрицательных — 150; остальные ошибочны. Вычислить precision и recall класса +1, сохранить в одноимённых переменных. Без вывода.

### Решение

```python
precision = 87 / 137
recall = 0.87
```

## 10. Повторение: матрица ошибок в многоклассовом уроке

**Задача на повторение / аналог предыдущей.**

Повторение: из 88 положительных объектов 81 предсказан как +1 и 7 как -1; из 45 отрицательных 40 предсказаны как -1 и 5 как +1. Сохранить TP,TN,FP,FN в одноимённых переменных. Вывод не нужен.

### Решение

```python
TP = 81
TN = 40
FP = 5
FN = 7
```

## 11. Повторение матрицы ошибок в уроке CART

**Задача на повторение / аналог предыдущей.**

Повторение: в тесте 25 положительных объектов (20 верных +1, 5 ошибок) и 75 отрицательных (60 верных -1, 15 ошибок). Вычислить TP,TN,FP,FN и сохранить в одноимённых переменных; без вывода.

### Решение

```python
# запишите вычисленные значения
TP = 20
TN = 60
FP = 15
FN = 5
```
