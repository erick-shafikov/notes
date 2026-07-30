# numpy — вероятность и статистика

## np.mean

Считает **среднее арифметическое** элементов: $\bar{x} = \frac{1}{N}\sum_{i=1}^N x_i$.

```python
np.mean(a, axis=None, dtype=None, keepdims=False)
```

- `axis=0` — среднее по строкам (результат — вектор средних по каждому столбцу)
- `axis=1` — среднее по столбцам (результат — вектор средних по каждой строке)
- без `axis` — скалярное среднее по всему массиву

```python
x = np.array([[1, 2], [3, 4], [5, 6]])

np.mean(x)          # 3.5  (все элементы)
np.mean(x, axis=0)  # [3.0, 4.0]  (среднее по каждому признаку)
np.mean(x, axis=1)  # [1.5, 3.5, 5.5]  (среднее по каждому объекту)
```

## np.var

Считает **дисперсию** — среднее квадратичное отклонение от среднего: $\sigma^2 = \frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^2$. Показывает, насколько сильно значения разбросаны вокруг среднего.

```python
np.var(a, axis=None, dtype=None, ddof=0, keepdims=False)
```

- По умолчанию `ddof=0` — смещённая оценка (делит на $N$): $\sigma^2 = \frac{1}{N}\sum(x_i - \bar{x})^2$
- `ddof=1` — несмещённая оценка (делит на $N-1$): $s^2 = \frac{1}{N-1}\sum(x_i - \bar{x})^2$
- `axis` работает так же, как в `np.mean`

```python
x = np.array([[1, 2], [3, 4], [5, 6]])

np.var(x, axis=0)         # [2.667, 2.667]  смещённая, по каждому столбцу
np.var(x, axis=0, ddof=1) # [4.0, 4.0]      несмещённая
```

**Паттерн Naive Bayes** — дисперсия по классу:

```python
Dx1, Dx2 = np.var(x_train[y_train == -1], axis=0)
# смещённая оценка (ddof=0) — стандарт для Gaussian Naive Bayes
```

> Sklearn's `GaussianNB` тоже использует `ddof=0` по умолчанию (`var_smoothing` добавляет малое $\epsilon$ для стабильности).

## np.random.multivariate_normal

```python
np.random.multivariate_normal(mean, cov, size=None)
```

Генерирует выборку из многомерного нормального распределения $\mathcal{N}(\mu, \Sigma)$.

- `mean` — вектор средних $\mu$, shape `(d,)`
- `cov` — ковариационная матрица $\Sigma$, shape `(d, d)`: симметричная, положительно полуопределённая
- `size` — количество точек; без `size` возвращает один вектор shape `(d,)`, при `size=n` — массив shape `(n, d)`

```python
mu    = [1.0, 2.0]
sigma = [[1.0, 0.8],
         [0.8, 1.0]]   # корреляция 0.8 между признаками

X = np.random.multivariate_normal(mu, sigma, size=500)
# X.shape == (500, 2)
```

### Ковариационная матрица

$\Sigma_{ij} = \mathrm{Cov}(x_i, x_j)$. На диагонали — дисперсии $\sigma_i^2$, вне диагонали — ковариации. Корреляция Пирсона: $\rho_{ij} = \Sigma_{ij} / (\sigma_i \sigma_j)$.

| Вид $\Sigma$              | Форма облака      | Пример                 |
| ------------------------- | ----------------- | ---------------------- |
| $\sigma^2 I$ (изотропная) | круг              | `[[1, 0], [0, 1]]`     |
| диагональная              | эллипс вдоль осей | `[[4, 0], [0, 1]]`     |
| полная                    | повёрнутый эллипс | `[[1, 0.8], [0.8, 1]]` |

```python
# Изотропная — два независимых признака с одинаковой дисперсией
X_iso = np.random.multivariate_normal([0, 0], np.eye(2), size=300)

# Разные дисперсии, нет корреляции
X_diag = np.random.multivariate_normal([0, 0], [[4, 0], [0, 1]], size=300)

# Сильная положительная корреляция
X_corr = np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=300)
```

### Применение в ML

**Генерация синтетических данных** с заданной структурой классов:

```python
n = 200
X_class0 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=n)
X_class1 = np.random.multivariate_normal(mean=[3, 3], cov=[[1, 0.5], [0.5, 1]], size=n)
X = np.vstack([X_class0, X_class1])
y = np.hstack([np.zeros(n), np.ones(n)])
```

**Gaussian Naive Bayes вручную** — сэмплинг из апостериорного:

```python
mu_c  = X[y == c].mean(axis=0)
cov_c = np.cov(X[y == c].T)          # несмещённая оценка, ddof=1
X_new = np.random.multivariate_normal(mu_c, cov_c, size=50)
```

### Построение $\Sigma$ из корреляций

Если известны $\sigma_i$ и $\rho_{ij}$:

```python
stds = np.array([2.0, 1.0])          # стандартные отклонения
corr = np.array([[1.0, 0.6],
                 [0.6, 1.0]])         # матрица корреляций
cov  = np.outer(stds, stds) * corr   # ковариационная матрица
```

### Восстановление параметров из выборки

```python
mu_hat  = X.mean(axis=0)     # оценка среднего
cov_hat = np.cov(X.T)        # оценка ковариационной матрицы (ddof=1)
```
