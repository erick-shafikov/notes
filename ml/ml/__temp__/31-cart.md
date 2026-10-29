# Деревья регрессии и CART

## 1. Impurity регрессионного разбиения

На сетке x от -2 до 3 с шагом 0.1 для f(x)=-x+0.2*x**2-0.5*sin(4*x)+cos(2*x) разделить объекты порогом t=0 на x<0 и x>=0. В каждой части прогнозом служит среднее y, impurity H — сумма квадратов отклонений от среднего. Сохранить HR1,HR2 и IG=H(R)-(|R1|/|R|)*H(R1)-(|R2|/|R|)*H(R2). Без вывода.

### Исходные данные и заготовка

```python
import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)
```

### Решение

```python
import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)

# здесь продолжайте программу

HR1 = np.var(y[x<0])*len(y[x<0]) 
HR2 = np.var(y[x>=0])*len(y[x>=0])
HR = np.var(y)*len(y)
IG = HR - HR1*sum(x<0)/len(x) - HR2*sum(x>=0)/len(x)
```

## 2. Лучший порог для дерева регрессии

Для f(x)=-x+0.2*x**2-0.5*sin(4*x)+cos(2*x) на исходной сетке [-2,3] перебрать пороги из x по правилу заготовки. Максимизировать IG с impurity, равным сумме квадратов отклонений y от среднего в части, и весами размеров частей. Сохранить порог th и выигрыш IG; не печатать.

### Исходные данные и заготовка

```python
import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)
```

### Решение

```python
import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)

def HR_(y):
    return np.square(y.mean() - y).sum()

IG = th = 0

for t in x:
    indx = x < t
    IG_ = HR_(y) - (np.sum(indx) * HR_(y[indx]) + np.sum(~indx) * HR_(y[~indx])) / len(y)
    if IG_ > IG:
        IG = IG_
        th = t
```

## 3. DecisionTreeRegressor

Аппроксимировать f(x)=0.3*x**2-0.2*x**3-0.5*sin(4*x) на сетке [-2,3] с шагом 0.1 деревом DecisionTreeRegressor с max_depth из заготовки. Обучить на x,y, предсказать на всей x. Результат сохранить в pr_y, MSE — в Q. Ничего не печатать.

### Исходные данные и заготовка

```python
import numpy as np
from sklearn import tree

x = np.arange(-2, 3, 0.1).reshape(-1, 1)
y = 0.3 * x ** 2 - 0.2 * x ** 3 - 0.5 * np.sin(4*x)
```

### Решение

```python
import numpy as np
from sklearn import tree

x = np.arange(-2, 3, 0.1).reshape(-1, 1)
y = 0.3 * x ** 2 - 0.2 * x ** 3 - 0.5 * np.sin(4*x)

clf = tree.DecisionTreeRegressor(max_depth=4)
clf.fit(x, y)

pr_y = clf.predict(x).reshape(-1, 1)

Q= ((pr_y - y) ** 2).mean()
```
