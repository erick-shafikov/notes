# sklearn.model_selection

## train_test_split

Делит выборку на обучающую и тестовую части.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # доля тестовой выборки (или int — число объектов)
    train_size=None,     # альтернатива test_size; если оба заданы — должны давать 1.0
    random_state=42,     # seed воспроизводимости
    shuffle=True,        # перемешать перед разбиением (по умолчанию True)
    stratify=y,          # сохранить пропорции классов (None — без стратификации)
)
```

`stratify=y` обязателен при несбалансированных классах — иначе train/test могут случайно получить разные пропорции.

```python
# только X, без y
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

# несколько массивов одновременно (разбиваются синхронно)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(X)), test_size=0.2
)
```

---

## cross_val_score

Кросс-валидация: делит данные на `cv` фолдов, обучает и оценивает модель на каждом.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    estimator,       # любой sklearn-совместимый классификатор/регрессор
    X, y,
    cv=5,            # число фолдов (int) или CV-объект (StratifiedKFold и т.п.)
    scoring='accuracy',  # метрика: 'accuracy', 'f1', 'roc_auc', 'r2', 'neg_mse' и др.
    n_jobs=-1,       # число параллельных процессов (-1 = все ядра)
)

print(scores)        # массив из cv значений метрики
print(scores.mean(), scores.std())
```

---

## GridSearchCV

Перебор гиперпараметров с кросс-валидацией.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}

gs = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    refit=True,       # после поиска переобучить лучшую модель на всех данных
    verbose=0,
)
gs.fit(X_train, y_train)

gs.best_params_       # {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'}
gs.best_score_        # средняя CV-метрика лучшей комбинации
gs.best_estimator_    # обученная лучшая модель (если refit=True)
gs.cv_results_        # детальные результаты — dict, удобно смотреть через pd.DataFrame
```

---

## StratifiedKFold

KFold с сохранением пропорций классов в каждом фолде — используется вместо простого `cv=5` при несбалансированных данных.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

# или передать в cross_val_score / GridSearchCV
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
```
