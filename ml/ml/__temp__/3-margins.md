# Отступы и ошибки классификации

## 1. Отступы и направление нормали

![Рисунок к задаче](assets/01de58368b474f7a.png)

```python
# Условие:
# Определить коэффициенты прямой по рисунку, направив вектор нормали в сторону объектов 1 и 2.
# Зелёные квадраты имеют метку +1, фиолетовые ромбы -1. Вычислить отступы объектов 1–6 и
# сохранить в margin, без вывода.
# Рисунок к условию: assets/01de58368b474f7a.png

import numpy as np

w = np.array([-16, -4, 9]) # задайте самостоятельно (параметры модели: w0, w1, w2 - ориентация разделяющей линии)
x_test = np.array(
    [[1, -5, 2], # 1
    [1, -4, 6], # 2
    [1, 3, 2],  # 3
    [1, 3, -3], # 4
    [1, 5, 6],  # 5
    [1, 9, 2],]  # 6
) # задайте самостоятельно (признаки образов: x0, x1, x2)

y_test = np.array([1, 1, 1, -1, -1, -1]) # задайте самостоятельно (метки класса)
margin = ((x_test @ w) * y_test).tolist()
```

## 2. Тестовые отступы и accuracy

```python
# Условие:
# Обучить линейный классификатор мини-батчевым SGD с экспоненциальной потерей exp(-M) на
# заданном разбиении. Сохранить коэффициенты в w. Вычислить отступы всех тестовых объектов,
# отсортировать по возрастанию и сохранить как np.array в mrgs. Accuracy тестовой выборки
# записать в acc. Настройки цикла из заготовки; вывод не нужен.

import numpy as np
from sklearn.model_selection import train_test_split

def loss(w, x, y):
    M = np.dot(w, x) * y
    return np.exp(-M)

def df(w, x, y):
    M = np.dot(w, x) * y
    return -np.exp(-M) * x * y

np.random.seed(0)

# Исходные параметры распределений двух классов
r1 = 0.4
D1 = 2.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 3.0
mean2 = [2, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# Моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.array([[1, x[0], x[1]] for x in np.hstack([x1, x2]).T])
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.3, shuffle=True)

n_train = len(x_train)  # Размер обучающей выборки
w = np.array([0.0, 0.0, 0.0])  # Начальные весовые коэффициенты
nt = np.array([0.5, 0.01, 0.01])
N = 500  # Число итераций алгоритма SGD
batch_size = 10  # Размер мини-батча (величина K = 10)

# Алгоритм стохастического градиентного спуска
for n in range(N):
    k = np.random.randint(0, n_train - batch_size)
    x_batch = x_train[k:k + batch_size]
    y_batch = y_train[k:k + batch_size]
    
    # Вычисление псевдоградиента
    grad = df(w, x_batch.T, y_batch).mean(axis=1)
    w -= nt * grad  # Обновление весов

# Вычисление отступов для тестовой выборки
mrgs = np.sort(x_test @ w * y_test)

# Вычисление метрики accuracy
predictions = np.sign(x_test @ w) # Предсказания по отступам
acc = np.mean(predictions == y_test)  # Точность предсказания
```
