# Метод опорных векторов

## 1. Линейный классификатор LinearSVC

Обучить svm.LinearSVC с настройками по умолчанию на заданной бинарной выборке. Из coef_[0] получить w=[w0,w1,w2] в соглашении исходной заготовки: для уравнения w0+w1*x1+w2*x2=0 первый коэффициент нужно умножить на w2. Сохранить w и количество неверных классификаций Q по знаку w·[1,x1,x2]. Без вывода.

### Исходные данные и заготовка

```python
import numpy as np
from sklearn import svm

data_x = [(4.9, 3.3), (5.6, 4.5), (6.4, 4.3), (6.7, 5.7), (6.3, 5.0), (5.2, 3.9), (5.5, 3.7), (5.6, 3.6), (5.5, 3.8), (6.1, 4.7), (7.4, 6.1), (6.0, 5.1), (5.5, 4.4), (5.9, 5.1), (6.5, 5.8), (6.5, 4.6), (6.7, 4.4), (6.3, 5.6), (5.9, 4.8), (6.0, 4.5), (5.6, 4.1), (5.6, 4.9), (4.9, 4.5), (6.2, 4.5), (6.1, 4.7), (6.1, 4.9), (6.2, 5.4), (5.7, 4.2), (6.1, 5.6), (5.8, 4.0), (6.6, 4.6), (5.6, 4.2), (7.2, 6.1), (7.7, 6.7), (5.6, 3.9), (7.7, 6.9), (6.0, 4.0), (6.1, 4.0), (7.6, 6.6), (5.1, 3.0), (6.3, 6.0), (6.7, 5.7), (6.8, 5.9), (6.4, 5.5), (7.0, 4.7), (5.8, 5.1), (5.8, 5.1), (6.4, 5.3), (6.3, 4.9), (6.4, 5.3), (5.7, 3.5), (7.2, 5.8), (6.4, 5.6), (5.7, 4.5), (6.0, 4.5), (7.7, 6.1), (6.2, 4.3), (7.1, 5.9), (7.3, 6.3), (5.0, 3.3), (6.3, 5.1), (5.8, 3.9), (6.4, 4.5), (6.3, 5.6), (6.8, 5.5), (6.9, 5.4), (5.5, 4.0), (5.7, 4.1), (6.5, 5.5), (6.3, 4.7), (5.0, 3.5), (6.7, 5.8), (6.9, 4.9), (7.7, 6.7), (5.8, 4.1), (6.4, 5.6), (6.7, 5.2), (6.7, 4.7), (5.4, 4.5), (6.8, 4.8), (5.7, 4.2), (5.5, 4.0), (6.3, 4.9), (6.5, 5.2), (5.8, 5.1), (6.0, 4.8), (6.2, 4.8), (6.5, 5.1), (7.9, 6.4), (6.7, 5.0), (6.7, 5.6), (6.0, 5.0), (6.1, 4.6), (5.7, 5.0), (7.2, 6.0), (6.3, 4.4), (5.9, 4.2), (6.9, 5.1), (6.6, 4.4), (6.9, 5.7)]
data_y = [-1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1]

# обучающая выборка в формате [1, x1, x2]
x_train = np.array([[1] + list(x) for x in data_x]) # входные образы
y_train = np.array(data_y) # целевые значения (метки классов)
```

### Решение

```python
import numpy as np
from sklearn import svm

data_x = [(4.9, 3.3), (5.6, 4.5), (6.4, 4.3), (6.7, 5.7), (6.3, 5.0), (5.2, 3.9), (5.5, 3.7), (5.6, 3.6), (5.5, 3.8), (6.1, 4.7), (7.4, 6.1), (6.0, 5.1), (5.5, 4.4), (5.9, 5.1), (6.5, 5.8), (6.5, 4.6), (6.7, 4.4), (6.3, 5.6), (5.9, 4.8), (6.0, 4.5), (5.6, 4.1), (5.6, 4.9), (4.9, 4.5), (6.2, 4.5), (6.1, 4.7), (6.1, 4.9), (6.2, 5.4), (5.7, 4.2), (6.1, 5.6), (5.8, 4.0), (6.6, 4.6), (5.6, 4.2), (7.2, 6.1), (7.7, 6.7), (5.6, 3.9), (7.7, 6.9), (6.0, 4.0), (6.1, 4.0), (7.6, 6.6), (5.1, 3.0), (6.3, 6.0), (6.7, 5.7), (6.8, 5.9), (6.4, 5.5), (7.0, 4.7), (5.8, 5.1), (5.8, 5.1), (6.4, 5.3), (6.3, 4.9), (6.4, 5.3), (5.7, 3.5), (7.2, 5.8), (6.4, 5.6), (5.7, 4.5), (6.0, 4.5), (7.7, 6.1), (6.2, 4.3), (7.1, 5.9), (7.3, 6.3), (5.0, 3.3), (6.3, 5.1), (5.8, 3.9), (6.4, 4.5), (6.3, 5.6), (6.8, 5.5), (6.9, 5.4), (5.5, 4.0), (5.7, 4.1), (6.5, 5.5), (6.3, 4.7), (5.0, 3.5), (6.7, 5.8), (6.9, 4.9), (7.7, 6.7), (5.8, 4.1), (6.4, 5.6), (6.7, 5.2), (6.7, 4.7), (5.4, 4.5), (6.8, 4.8), (5.7, 4.2), (5.5, 4.0), (6.3, 4.9), (6.5, 5.2), (5.8, 5.1), (6.0, 4.8), (6.2, 4.8), (6.5, 5.1), (7.9, 6.4), (6.7, 5.0), (6.7, 5.6), (6.0, 5.0), (6.1, 4.6), (5.7, 5.0), (7.2, 6.0), (6.3, 4.4), (5.9, 4.2), (6.9, 5.1), (6.6, 4.4), (6.9, 5.7)]
data_y = [-1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1]

# обучающая выборка в формате [1, x1, x2]
x_train = np.array([[1] + list(x) for x in data_x]) # входные образы
# x_train = np.array(data_x)
y_train = np.array(data_y) # целевые значения (метки классов)

# Создаем объект линейного классификатора (Linear Support Vector Classification)
lin_clf = svm.LinearSVC()

# Обучаем и извлекаем веса
w = lin_clf.fit(x_train, y_train).coef_[0]

# Переписываем w_0 на правильный
w[0] *= w[2]

# Качество
Q = sum(np.sign(x_train @ w) != y_train)

# Визуализация данных и разделяющих прямых
import matplotlib.pyplot as plt

# Отображаем точки данных
[plt.scatter(x[1], x[2], color='red', marker='o') for x in x_train[y_train == -1]]
[plt.scatter(x[1], x[2], color='blue', marker='o') for x in x_train[y_train == 1]]

# Определяем границы разделения для w
x_range = np.linspace(4, 8) # диапазон по оси x
y_range_w = -(w[1] * x_range + w[0]) / w[2] # уравнение разделяющей прямой для w

# Отображаем границы разделения
plt.plot(x_range, y_range_w, color='green', label='Граница')

# Настройки графика
plt.xlim(4.5, 8)
plt.ylim(2, 8)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear SVC')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0,color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
```

## 2. SVC и опорные векторы

Обучить svm.SVC(kernel="linear") именно на data_x,data_y. Получить w=[intercept_[0],*coef_[0]], опорные векторы v_support и число ошибок Q по знаку w·[1,x1,x2] на обучающей выборке. Сохранить все три результата; не печатать.

### Исходные данные и заготовка

```python
v_support = clf.support_vectors_ # список опорных векторов
```

### Решение

```python
import numpy as np
from sklearn import svm

data_x = [(3.0, 4.9), (2.7, 3.9), (3.0, 5.5), (2.6, 4.0), (2.9, 4.3), (3.1, 5.1), (2.2, 4.5), (2.3, 3.3), (2.7, 5.1), (3.3, 5.7), (2.8, 5.1), (2.8, 4.9), (2.5, 4.5), (2.8, 4.7), (3.2, 4.7), (3.2, 5.7), (2.8, 6.1), (3.6, 6.1), (2.8, 4.8), (2.9, 4.5), (3.1, 4.9), (2.3, 4.4), (3.3, 6.0), (2.6, 5.6), (3.0, 4.4), (2.9, 4.7), (2.8, 4.0), (2.5, 5.8), (2.4, 3.3), (2.8, 6.7), (3.0, 5.1), (2.3, 4.0), (3.1, 5.5), (2.8, 4.8), (2.7, 5.1), (2.5, 4.0), (3.1, 4.4), (3.8, 6.7), (3.1, 5.6), (3.1, 4.7), (3.0, 5.8), (3.0, 5.2), (3.0, 4.5), (2.7, 4.9), (3.0, 6.6), (2.9, 4.6), (3.0, 4.6), (2.6, 3.5), (2.7, 5.1), (2.5, 5.0), (2.0, 3.5), (3.2, 5.9), (2.5, 5.0), (3.4, 5.6), (3.4, 4.5), (3.2, 5.3), (2.2, 4.0), (2.2, 5.0), (3.3, 4.7), (2.7, 4.1), (2.4, 3.7), (3.0, 4.2), (3.2, 6.0), (3.0, 4.2), (3.0, 4.5), (2.7, 4.2), (2.5, 3.0), (2.8, 4.6), (2.9, 4.2), (3.1, 5.4), (2.5, 4.9), (3.2, 5.1), (2.8, 4.5), (2.8, 5.6), (3.4, 5.4), (2.7, 3.9), (3.0, 6.1), (3.0, 5.8), (3.0, 4.1), (2.5, 3.9), (2.4, 3.8), (2.6, 4.4), (2.9, 3.6), (3.3, 5.7), (2.9, 5.6), (3.0, 5.2), (3.0, 4.8), (2.7, 5.3), (2.8, 4.1), (2.8, 5.6), (3.2, 4.5), (3.0, 5.9), (2.9, 4.3), (2.6, 6.9), (2.8, 5.1), (2.9, 6.3), (3.2, 4.8), (3.0, 5.5), (3.0, 5.0), (3.8, 6.4)]
data_y = [1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1]

# Создаём объект класса SVM (Support Vector Machine)
clf = svm.SVC(kernel='linear').fit(data_x, data_y)
w = [clf.intercept_[0], clf.coef_[0][0], clf.coef_[0][1]]
v_support = clf.support_vectors_

# Качество
Q = sum([np.sign(w[0] + w[1] * x[0] + w[2] * x[1]) != y for x, y in zip(data_x, data_y)])

# Визуализация
import matplotlib.pyplot as plt

# Точки данных
[plt.scatter(x[0], x[1], color='red', marker='o') for x, y in zip(data_x, data_y) if y == -1]
[plt.scatter(x[0], x[1], color='blue', marker='o') for x, y in zip(data_x, data_y) if y == 1]

# Линия
x_range = np.linspace(1.5, 8) # диапазон по оси x1
y_range_w = -(w[1] * x_range + w[0]) / w[2] # значения по x2 для гиперплоскости
plt.plot(x_range, y_range_w, color='green', label='Граница')

# Настройки графика
plt.xlim(1.5, 4)
plt.ylim(2, 8)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear SVC')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0,color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
```

## 3. Регрессия SVR

С помощью линейного svm.SVR аппроксимировать f(x)=sin(x/2)+0.2*cos(2*x)-0.1*sin(4*x)-2.5 моделью с признаками [1,x,x**2,x**3,cos(x),sin(x)]. Использовать настройки SVR и разбиение из заготовки. Коэффициенты coef_ и смещение intercept_ собрать в w согласно заданной model; вычислить среднюю квадратичную ошибку Q. Без вывода.

### Исходные данные и заготовка

```python
import numpy as np
from sklearn import svm

def func(x):
    return np.sin(0.5*x) + 0.2 * np.cos(2*x) - 0.1 * np.sin(4 * x) - 2.5

def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3 + w[4] * np.cos(x) + w[5] * np.sin(x)

# обучающая выборка
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

x_train = np.array([[x, x**2, x**3, np.cos(x), np.sin(x)] for x in coord_x])
y_train = coord_y
```

### Решение

```python
import numpy as np
from sklearn import svm

def func(x):
    return np.sin(0.5*x) + 0.2 * np.cos(2*x) - 0.1 * np.sin(4 * x) - 2.5


def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3 + w[4] * np.cos(x) + w[5] * np.sin(x)


# обучающая выборка
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

x_train = np.array([[x, x**2, x**3, np.cos(x), np.sin(x)] for x in coord_x])
y_train = coord_y

# здесь продолжайте программу
svr = svm.SVR(kernel='linear')    # SVM с линейным ядром
svr.fit(x_train, y_train)           # нахождение вектора w по обучающей выборке

w = np.array([svr.intercept_[0], *svr.coef_[0]])
predict = model(w, coord_x)
Q = np.mean((y_train - predict)**2)
```

## 4. Регрессия с радиальным ядром

На [-4,6] аппроксимировать f(x)=sin(x/2)+0.2*cos(2*x)-0.1*sin(4*x)+3 с помощью SVR с радиальным ядром и настройками из заготовки. Обучить только по каждому третьему отсчёту coord_x,coord_y; предсказать все отсчёты. Сохранить predict и MSE по всей сетке в Q. Не печатать.

### Исходные данные и заготовка

```python
x_train = coord_x[::3]
y_train = coord_y[::3]
```

### Решение

```python
import numpy as np
from sklearn import svm
#import matplotlib.pyplot as plt


def func(x):
    return np.sin(0.5*x) + 0.2 * np.cos(2*x) - 0.1 * np.sin(4 * x) + 3


# обучающая выборка
coord_x = np.expand_dims(np.arange(-4.0, 6.0, 0.1), axis=1)
coord_y = func(coord_x).ravel()

# здесь продолжайте программу
svr = svm.SVR(kernel="rbf")

x_train = coord_x[::3]
y_train = coord_y[::3]

svr.fit(x_train, y_train)
predict = svr.predict(coord_x)

Q = ((predict - coord_y)**2).mean()
#plt.plot(coord_y)
#plt.plot(svr.predict(coord_x))
```

## 5. Нелинейная бинарная классификация

По сгенерированным двум классам {-1,+1} обучить SVC с радиальным ядром, используя настройки из заготовки. Оценить долю ошибок на x_test,y_test и сохранить её в Q. Самостоятельно сравнить с линейным ядром; вывода не требуется.

### Решение

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.4, shuffle=True)

# Обучение двух моделей SVM (линейное и радиальное ядра) для сравнения графиков
clf_linear = svm.SVC(kernel='linear').fit(x_train, y_train)
clf_rbf = svm.SVC(kernel='rbf').fit(x_train, y_train)

predict = clf_rbf.predict(x_test)

Q = (predict != y_test).mean()

'''ГРАФИКИ'''
# Функция для построения области решений
def plot_decision_boundary(ax, clf, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', s=20)
    
    predictions = clf.predict(X)
    errors = (predictions != y).sum()
    accuracy = accuracy_score(y, predictions)
    
    ax.set_title(f"{title}\nНеправильные классификации: {errors}\nТочность: {accuracy:.2f}")
    ax.set_xlabel("Признак 1")
    ax.set_ylabel("Признак 2")

# Построение графиков
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# График для линейного ядра
plot_decision_boundary(axes[0], clf_linear, x_test, y_test, "SVM с линейным ядром")

# График для радиального ядра
plot_decision_boundary(axes[1], clf_rbf, x_test, y_test, "SVM с радиальным ядром")

# Легенда
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f2d5c9', markersize=8, label='Класс -1'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e8f7ff', markersize=8, label='Класс 1')]
for ax in axes:
    ax.legend(handles=handles, loc='upper right')

plt.tight_layout()
plt.show()
```
