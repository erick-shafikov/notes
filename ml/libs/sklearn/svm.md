# Классификация

## SVC

```python
from sklearn.svm import SVC

clf = SVC(
  C=1.0,                        # параметр регуляризации: меньше C — шире отступ, больше ошибок; больше C — жёсткая граница
  kernel='rbf',                 # ядро: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
  degree=3,                     # степень полинома для kernel='poly', для других игнорируется
  gamma='scale',                # коэф. ядра для 'rbf','poly','sigmoid': 'scale'=1/(n_features*X.var()), 'auto'=1/n_features
  coef0=0.0,                    # свободный член в ядре — значим только для 'poly' и 'sigmoid'
  shrinking=True,               # использовать ли эвристику сжатия (ускоряет обучение)
  probability=False,            # включить оценку вероятностей (замедляет fit, нужен для predict_proba)
  tol=1e-3,                     # допуск критерия останова
  cache_size=200,               # размер кэша ядра в MB
  class_weight=None,            # веса классов: dict {класс: вес} или 'balanced'
  verbose=False,                # выводить ли прогресс оптимизатора (не потокобезопасно)
  max_iter=-1,                  # максимум итераций оптимизатора, -1 = без ограничений
  decision_function_shape='ovr',# форма выходов многоклассовой задачи: 'ovr' (one-vs-rest) или 'ovo' (one-vs-one)
  break_ties=False,             # разрывать ли ничьи по уверенности (только ovr + >2 классов)
  random_state=None,            # seed для перемешивания при probability=True
)
clf.fit(X, y)
clf.predict(X_test)

# атрибуты
clf.class_weight_       # итоговые веса классов, shape (n_classes,)
clf.classes_            # уникальные метки классов, shape (n_classes,)
clf.coef_               # веса признаков — только при kernel='linear', shape (n_classes*(n_classes-1)/2, n_features)
clf.dual_coef_          # двойственные коэф. support vectors * их метки, shape (n_classes-1, n_SV)
clf.fit_status_         # 0 — сошлось, 1 — не сошлось (с предупреждением)
clf.intercept_          # свободный член (смещение гиперплоскости), shape (n_classes*(n_classes-1)/2,)
clf.n_features_in_      # число признаков при обучении
clf.feature_names_in_   # имена признаков (если X был DataFrame)
clf.n_iter_             # число итераций оптимизатора по каждой паре классов
clf.support_            # индексы support vectors в обучающей выборке, shape (n_SV,)
clf.support_vectors_    # сами support vectors, shape (n_SV, n_features)
clf.n_support_          # число support vectors на каждый класс, shape (n_classes,)
clf.probA_              # коэф. A Platt scaling (только при probability=True)
clf.probB_              # коэф. B Platt scaling (только при probability=True)

# [1] Platt scaling: P(y=1|x) = 1 / (1 + exp(A·f + B)), где f = decision_function(x)
#     probA_/probB_ — это A и B, подобранные внутри fit() через 5-кратную кросс-валидацию
#     сами вероятности возвращает predict_proba(X)
clf.shape_fit_          # форма обучающего массива X

# методы
clf.decision_function(X)   # расстояние до разделяющей гиперплоскости, shape (n_samples, n_classes)
clf.fit(X, y)              # обучение модели
clf.get_metadata_routing() # маршрутизация метаданных для Pipeline
clf.get_params()           # получить словарь гиперпараметров, то есть заданные аргументы SVC
clf.predict(X)             # предсказать метки классов
clf.predict_log_proba(X)   # лог-вероятности классов (только при probability=True)
clf.predict_proba(X)       # вероятности классов (только при probability=True)
clf.score(X, y)            # accuracy
clf.set_fit_request()      # настроить metadata routing для fit
clf.set_params(**params)   # установить гиперпараметры
clf.set_score_request()    # настроить metadata routing для score
```

## NuSVC

За основу — SVC (libsvm). `C` заменён на `nu`.

```python
from sklearn.svm import NuSVC

clf = NuSVC(
  nu=0.5,                        # ВМЕСТО C: (0,1] — верхняя граница доли ошибок и нижняя граница доли support vectors
  kernel='rbf', degree=3, gamma='scale', coef0=0.0,
  shrinking=True, tol=1e-3, cache_size=200, class_weight=None,
  verbose=False, max_iter=-1, decision_function_shape='ovr',
  break_ties=False, random_state=None,
)
# нет: probability

clf.fit(X, y)
clf.predict(X_test)

# атрибуты — идентичны SVC

# методы — идентичны SVC
# predict_proba, predict_log_proba — deprecated в 1.9, используй CalibratedClassifierCV(NuSVC())
```

## LinearSVC

За основу — SVC, но через **liblinear**: только линейное ядро, быстрее масштабируется.

```python
from sklearn.svm import LinearSVC

clf = LinearSVC(
  penalty='l2',          # норма регуляризации: 'l2' или 'l1' (→ разреженные веса)
  loss='squared_hinge',  # 'hinge' (стандартный SVM) или 'squared_hinge'
  dual='auto',           # 'auto'/True/False — двойственная или прямая задача; False предпочтительнее при n_samples > n_features
  multi_class='ovr',     # стратегия многоклассовой задачи: 'ovr' или 'crammer_singer'
  fit_intercept=True,    # подбирать ли свободный член
  intercept_scaling=1,   # масштаб синтетического признака для intercept (снижает влияние регуляризации на него)
  C=1.0, tol=1e-4, class_weight=None, verbose=0, random_state=None, max_iter=1000,
)
# нет: kernel, degree, gamma, coef0, shrinking, probability, cache_size, decision_function_shape, break_ties

clf.fit(X, y)
clf.predict(X_test)

# атрибуты
clf.coef_             # shape (1, n_features) при 2 классах или (n_classes, n_features)
clf.intercept_        # shape (1,) или (n_classes,)
clf.classes_
clf.n_features_in_
clf.feature_names_in_
clf.n_iter_
# нет: dual_coef_, fit_status_, support_, support_vectors_, n_support_, probA_, probB_, shape_fit_, class_weight_

# методы
clf.densify()   # coef_ sparse → dense ndarray
clf.sparsify()  # coef_ dense → sparse (экономит память при L1-регуляризации)
# нет: predict_proba, predict_log_proba
```

# Регрессия

## SVR

За основу — SVC (libsvm), регрессионный вариант. Предсказывает непрерывные значения.

```python
from sklearn.svm import SVR

regr = SVR(
  epsilon=0.1,  # ширина epsilon-трубки — точки внутри не штрафуются
  C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
  shrinking=True, tol=1e-3, cache_size=200, verbose=False, max_iter=-1,
)
# нет: probability, class_weight, decision_function_shape, break_ties, random_state

regr.fit(X, y)
regr.predict(X_test)  # вещественные значения

# атрибуты — как у SVC
# нет: class_weight_, classes_, probA_, probB_

# методы
# score(X, y) — возвращает R² вместо accuracy
# нет: decision_function, predict_proba, predict_log_proba
```

## NuSVR

За основу — SVR (libsvm). `epsilon` заменён на `nu`; `C` сохраняется.

```python
from sklearn.svm import NuSVR

regr = NuSVR(
  nu=0.5,  # ВМЕСТО epsilon: (0,1] — верхняя граница доли ошибок и нижняя граница доли support vectors
  C=1.0,   # C остаётся (в NuSVC его нет)
  kernel='rbf', degree=3, gamma='scale', coef0=0.0,
  shrinking=True, tol=1e-3, cache_size=200, verbose=False, max_iter=-1,
)

# атрибуты, методы — идентичны SVR
```

## LinearSVR

За основу — SVR, но через **liblinear**: только линейное ядро, быстрее масштабируется.

```python
from sklearn.svm import LinearSVR

regr = LinearSVR(
  epsilon=0.0,                  # ширина epsilon-трубки
  loss='epsilon_insensitive',   # 'epsilon_insensitive' (L1-потеря) или 'squared_epsilon_insensitive' (L2-потеря)
  fit_intercept=True,           # подбирать ли свободный член
  intercept_scaling=1.0,        # масштаб синтетического признака для intercept
  dual='auto',                  # 'auto'/True/False — тип задачи оптимизации
  C=1.0, tol=1e-4, verbose=0, random_state=None, max_iter=1000,
)
# нет: kernel, degree, gamma, coef0, shrinking, cache_size

regr.fit(X, y)
regr.predict(X_test)

# атрибуты
regr.coef_
regr.intercept_
regr.n_features_in_
regr.feature_names_in_
regr.n_iter_
# нет: dual_coef_, fit_status_, n_support_, shape_fit_, support_, support_vectors_

# методы — идентичны SVR
```

# Обнаружение аномалий

## OneClassSVM

За основу — SVC (libsvm). Unsupervised: обучается без меток, возвращает +1 (норма) / -1 (аномалия).

```python
from sklearn.svm import OneClassSVM

clf = OneClassSVM(
  nu=0.5,  # ВМЕСТО C: верхняя граница ожидаемой доли выбросов в обучающей выборке
  kernel='rbf', degree=3, gamma='scale', coef0=0.0,
  shrinking=True, tol=1e-3, cache_size=200, verbose=False, max_iter=-1,
)
# нет: probability, class_weight, decision_function_shape, break_ties, random_state

clf.fit(X)           # без y — unsupervised
clf.predict(X_test)  # +1 (inlier / норма) или -1 (outlier / аномалия)

# атрибуты
clf.offset_  # сдвиг: decision_function = score_samples - offset_
# нет: class_weight_, classes_, probA_, probB_

# методы
clf.fit(X)                # без y
clf.fit_predict(X)        # fit + predict за один вызов, возвращает +1/-1
clf.score_samples(X)      # raw scores без сдвига на offset_
# decision_function(X) — >0 норма, <0 аномалия (изменена семантика)
# нет: predict_proba, predict_log_proba
```

---

## Что происходит под капотом

Задача SVM — **квадратичное программирование** (QP): целевая функция `½||ω||²` квадратична, ограничения линейны. Решать QP напрямую дорого — O(n³) по памяти и времени.

**libsvm** (SVC, SVR, NuSVC, NuSVR, OneClassSVM) использует **SMO (Sequential Minimal Optimization)**, алгоритм Платта (1998):

- Вместо оптимизации всех `n` переменных `λᵢ` сразу — на каждом шаге выбираются **ровно 2 переменные** и оптимизируются при фиксированных остальных
- Минимум 2 переменные: ограничение `Σ λᵢyᵢ = 0` связывает переменные, и 2 — это минимум, при котором подзадача допустима
- Подзадача из 2 переменных решается **аналитически**, без численного решателя
- Алгоритм повторяет шаги до сходимости

**liblinear** (LinearSVC, LinearSVR) использует **coordinate descent** по прямой или двойственной задаче — быстрее при линейном ядре на больших данных.

---

## Сравнительная справка

| Класс         | Задача        | Бэкенд    | Ядра          | Ключевой параметр | Масштабируемость |
| ------------- | ------------- | --------- | ------------- | ----------------- | ---------------- |
| `SVC`         | Классификация | libsvm    | все           | `C`               | средняя          |
| `NuSVC`       | Классификация | libsvm    | все           | `nu`              | средняя          |
| `LinearSVC`   | Классификация | liblinear | только linear | `C`               | высокая          |
| `SVR`         | Регрессия     | libsvm    | все           | `C`, `epsilon`    | средняя          |
| `NuSVR`       | Регрессия     | libsvm    | все           | `nu`, `C`         | средняя          |
| `LinearSVR`   | Регрессия     | liblinear | только linear | `C`, `epsilon`    | высокая          |
| `OneClassSVM` | Аномалии      | libsvm    | все           | `nu`              | средняя          |

**Когда что применять:**

- **`SVC` / `SVR`** — стандартный выбор для небольших датасетов (до ~10k), нужно нелинейное ядро.
- **`NuSVC` / `NuSVR`** — то же что SVC/SVR, но удобнее: `nu` ∈ (0, 1] имеет прямой смысл (доля support vectors / ошибок), проще интерпретировать чем подобранный `C`.
- **`LinearSVC` / `LinearSVR`** — большие датасеты (>10k), линейная граница. liblinear значительно быстрее libsvm при линейном ядре. `LinearSVC` поддерживает L1-регуляризацию → разреженные веса.
- **`OneClassSVM`** — novelty / anomaly detection. Обучается только на нормальных данных, предсказывает +1 (норма) или -1 (аномалия) для новых точек.
