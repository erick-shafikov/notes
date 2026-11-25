# sklearn.cluster

## DBSCAN

Плотностная кластеризация: объединяет точки, у которых достаточно соседей в радиусе `eps`, и помечает точки в разреженных областях как шум. Число кластеров заранее задавать не нужно; кластеры могут иметь произвольную форму.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.5,                 # максимальное расстояние до соседа; граница включается: distance <= eps
    min_samples=5,           # минимум точек в eps-окрестности для корневой точки, включая саму точку
    metric='euclidean',      # метрика расстояния; также callable или 'precomputed'
    metric_params=None,      # дополнительные параметры метрики
    algorithm='auto',        # поиск соседей: 'auto', 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,            # размер листа BallTree/KDTree; влияет на скорость и память
    p=None,                  # степень Минковского; None означает p=2
    n_jobs=None,             # число потоков для поиска соседей; -1 = все ядра
)

labels = dbscan.fit_predict(X)  # shape (n_samples,); кластеры: 0, 1, ..., шум: -1
```

Точка считается:

- **корневой**, если в её `eps`-окрестности не меньше `min_samples` точек с учётом её самой;
- **граничной**, если она не корневая, но входит в окрестность корневой точки;
- **шумовой**, если не относится ни к одному кластеру (`label == -1`).

```python
# атрибуты после fit
dbscan.labels_                 # метка каждой обучающей точки; шум имеет метку -1
dbscan.core_sample_indices_    # индексы корневых точек в исходном X
dbscan.components_             # копии корневых точек, shape (n_core_samples, n_features)
dbscan.n_features_in_          # число признаков, увиденных при fit
dbscan.feature_names_in_       # имена признаков, если X — DataFrame со строковыми колонками
```

```python
# методы
dbscan.fit(X)                                  # построить кластеры; результат находится в labels_
dbscan.fit(X, sample_weight=weights)           # учитывать веса объектов
dbscan.fit_predict(X)                          # fit + вернуть labels_
dbscan.fit_predict(X, sample_weight=weights)   # то же с весами
dbscan.get_params()                            # получить гиперпараметры
dbscan.set_params(eps=0.7, min_samples=8)      # изменить гиперпараметры
```

У `DBSCAN` нет `predict(X_new)`: модель строит связные плотные области только для данных, переданных в `fit`. Для новых точек нужно заново кластеризовать объединённую выборку либо отдельно реализовать присвоение по ближайшим корневым точкам.

### Пример

Расстояния зависят от масштаба признаков, поэтому числовые признаки обычно стандартизируют перед DBSCAN.

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

model = DBSCAN(eps=0.35, min_samples=5)
labels = model.fit_predict(X_scaled)

clusters = sorted(set(labels) - {-1})
n_clusters = len(clusters)
n_noise = np.count_nonzero(labels == -1)

X_cluster_0 = X[labels == 0]
X_noise = X[labels == -1]
```

### Выбор параметров

- Уменьшение `eps` даёт больше мелких кластеров и шума; увеличение — меньше крупных кластеров и может слить разные группы.
- Увеличение `min_samples` требует более плотных областей и обычно увеличивает количество шума.
- Для выбора `eps` можно построить график расстояния до `k`-го ближайшего соседа, где `k = min_samples`, и искать «локоть».
- Один `eps` плохо работает, если кластеры имеют сильно разную плотность; в таком случае стоит рассмотреть `OPTICS`.

### Предвычисленные расстояния

При `metric='precomputed'` вместо признаков передаётся квадратная матрица расстояний:

```python
from sklearn.metrics import pairwise_distances

D = pairwise_distances(X, metric='cosine')
labels = DBSCAN(
    eps=0.2,
    min_samples=5,
    metric='precomputed',
).fit_predict(D)
```

Реализация scikit-learn заранее вычисляет окрестности и при большом `eps` и малом `min_samples` может потребовать до $O(n^2)$ памяти.
