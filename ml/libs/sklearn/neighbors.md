# KNeighborsClassifier

```python
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(
  n_neighbors=5,                # k — число соседей, участвующих в голосовании
  weights='uniform',            # веса соседей: 'uniform' — равные, 'distance' — обратно пропорциональны расстоянию, либо callable(distances) -> веса того же shape
  algorithm='auto',             # поиск соседей: 'auto', 'ball_tree', 'kd_tree', 'brute'; 'auto' выбирает по данным (на разреженных или при metric='precomputed' — brute)
  leaf_size=30,                 # размер листа для ball_tree/kd_tree: влияет на скорость построения, запроса и память
  p=2,                          # степень метрики Минковского: 1 — манхэттенская (L1), 2 — евклидова (L2); только при metric='minkowski'
  metric='minkowski',           # метрика расстояния: строка ('euclidean', 'manhattan', 'chebyshev', 'cosine', ...), 'precomputed' или callable(x1, x2)
  metric_params=None,           # доп. аргументы для метрики (dict), например {'V': cov} для 'mahalanobis'
  n_jobs=None,                  # число потоков для поиска соседей, -1 = все ядра; на fit не влияет
)
clf.fit(X, y)                   # X: (n_samples, n_features), y: (n_samples,); «обучение» — запоминание выборки и построение индекса (дерева)
clf.predict(X_test)             # метки классов, shape (n_test,)
```

```python
# атрибуты
clf.classes_                    # уникальные метки классов (отсортированы), shape (n_classes,)
clf.effective_metric_           # реально используемая метрика (если p=2 и metric='minkowski' → 'euclidean')
clf.effective_metric_params_    # реально используемые доп. параметры метрики
clf.n_features_in_              # число признаков при обучении
clf.feature_names_in_           # имена признаков (если X был DataFrame)
clf.n_samples_fit_              # число объектов в обучающей выборке
clf.outputs_2d_                 # True, если y при fit был 2D (multi-output), иначе False
```

```python
# методы
clf.fit(X, y)                                       # запомнить выборку и построить структуру поиска
clf.predict(X)                                      # метка по взвешенному большинству среди k соседей
clf.predict_proba(X)                                # доля (взвешенных) голосов за каждый класс, shape (n_samples, n_classes); порядок столбцов как в classes_
clf.score(X, y)                                     # accuracy
clf.kneighbors(X=None, n_neighbors=None, return_distance=True)
# возвращает (distances, indices), оба shape (n_samples, n_neighbors), отсортированы по возрастанию расстояния; indices — позиции в обучающей выборке
# если X=None — ищет соседей для самих обучающих объектов, при этом сам объект исключается (аналог leave-one-out)
clf.kneighbors_graph(X=None, n_neighbors=None, mode='connectivity')
# разреженная матрица (n_samples, n_samples_fit): mode='connectivity' — 0/1, mode='distance' — расстояния
clf.get_params()                                    # словарь гиперпараметров
clf.set_params(**params)                            # установить гиперпараметры
clf.get_metadata_routing()                          # маршрутизация метаданных для Pipeline
clf.set_score_request()                             # настроить metadata routing для score
```
