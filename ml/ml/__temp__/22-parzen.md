# Парзеновское окно и потенциальные функции

## 1. Классификация с гауссовским парзеновским окном

Классифицировать x_test трёх классов по сумме весов обучающих объектов каждого класса. Расстояние — сумма абсолютных разностей признаков, ядро K(r)=exp(-r**2/2)/sqrt(2*pi), ширина h=1. Для каждого объекта выбрать класс с максимальной суммой K(distance/h). Сохранить predict и долю ошибок Q. Не печатать.

### Исходные данные и заготовка

```python
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(0)
n_feature = 2 # число признаков

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 3]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [1, 1] # + np.array(range(1, n_feature+1)) * 0.5
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r3 = -0.7
D3 = 1.0
mean3 = [-2, -2]  # + np.array(range(1, n_feature+1)) * -0.5
V3 = [[D3 * r3 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N1, N2, N3 = 200, 150, 190
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
x3 = np.random.multivariate_normal(mean3, V3, N3).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N1), np.ones(N2), np.ones(N3) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.5, shuffle=True)
```

### Решение

```python
import numpy as np
from sklearn.model_selection import train_test_split


np.random.seed(0)
n_feature = 2 # число признаков

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 3]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [1, 1] # + np.array(range(1, n_feature+1)) * 0.5
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r3 = -0.7
D3 = 1.0
mean3 = [-2, -2]  # + np.array(range(1, n_feature+1)) * -0.5
V3 = [[D3 * r3 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N1, N2, N3 = 200, 150, 190
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T
x3 = np.random.multivariate_normal(mean3, V3, N3).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N1), np.ones(N2), np.ones(N3) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123, test_size=0.5, shuffle=True)

# здесь продолжайте программу
class ParzenWindow:      
    def __init__(self, *, h: int):
        self.h = h
    
    @staticmethod
    def get_gausian(r: np.array):
        return 1/np.sqrt(2*np.pi) * np.exp(-r**2/2)
        
    
    def fit(self, X: np.array, y: np.array) -> None:
        self._X_train = X
        self._y_train = y
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
    def predict(self, X_pred: np.array) -> int:
        pred_cls = []
        
        for i in X_pred:
            r = abs(i - self._X_train).sum(axis=1) / self.h
            distances = self.get_gausian(r)
            hstack = np.stack([distances, self._y_train], axis=1)
            
            sum_distances = {c: hstack[hstack[:, 1]==c][:, 0].sum() for c in self.classes_}
            
            pred = max(sum_distances, key=sum_distances.get)
            pred_cls.append(pred)     
                        
        return pred_cls

h = 1
     
pw = ParzenWindow(h=h)
pw.fit(x_train, y_train)
predict = pw.predict(x_test)

Q = (predict!=y_test).mean()
```
