# sklearn.decomposition

## PCA

```python
from sklearn.decomposition import PCA

pca = PCA(
    n_components=2,        # число компонент (int) или доля дисперсии (float 0–1)
    whiten=False,          # нормировать компоненты до unit variance
    svd_solver='auto',     # 'auto', 'full', 'arpack', 'randomized'
    random_state=42,
)

# fit вычисляет главные компоненты по обучающей выборке
pca.fit(X_train)

# transform проецирует данные в новое пространство
X_reduced = pca.transform(X_train)     # (n_samples, n_components)
X_reduced = pca.fit_transform(X_train) # fit + transform за один вызов

# обратное преобразование (приближённое восстановление)
X_approx = pca.inverse_transform(X_reduced)

# атрибуты
pca.components_              # (n_components, n_features) — главные направления (строки = векторы)
pca.explained_variance_      # дисперсия вдоль каждой компоненты
pca.explained_variance_ratio_ # доля объяснённой дисперсии каждой компоненты
pca.singular_values_         # сингулярные значения (связаны: σ_j² = explained_variance_[j] * (n-1))
pca.mean_                    # среднее по обучающей выборке (вычитается перед SVD)
pca.n_components_            # фактическое число компонент после fit
```

`n_components=0.95` — автоматически выбирает минимальное число компонент, объясняющих 95% дисперсии.

Внутри PCA использует SVD (через `np.linalg.svd` или randomized), не `eig(X^T X)` — численно устойчивее.

---

## TruncatedSVD

Работает с разреженными матрицами (sparse) — в отличие от PCA. Применяется для текстов (TF-IDF матрицы), рекомендательных систем.

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(
    n_components=2,        # число компонент (обязательный параметр)
    algorithm='randomized',# 'randomized' (быстро) или 'arpack' (точнее)
    n_iter=5,              # число итераций randomized SVD
    random_state=42,
)

X_reduced = svd.fit_transform(X)    # (n_samples, n_components)
X_approx = svd.inverse_transform(X_reduced)

# атрибуты — как у PCA
svd.components_               # (n_components, n_features)
svd.explained_variance_       # дисперсия каждой компоненты
svd.explained_variance_ratio_ # доля объяснённой дисперсии
svd.singular_values_          # сингулярные значения σ_j
```

**Отличие от PCA**: TruncatedSVD не центрирует данные (не вычитает среднее). На центрированных данных результаты совпадают с PCA.

---

## Сравнение

| | PCA | TruncatedSVD |
|--|-----|-------------|
| Sparse input | нет (требует dense) | да |
| Центрирование | да (автоматически) | нет |
| Применение | числовые признаки | тексты, разреженные матрицы |
| Solver | полный или randomized SVD | randomized / arpack |
