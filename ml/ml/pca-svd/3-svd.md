# SVD — Singular Value Decomposition (сингулярное разложение)

Любую матрицу $X \in \mathbb{R}^{m \times n}$ можно разложить в произведение трёх матриц:

$$X = U \Sigma V^T$$

- $U \in \mathbb{R}^{m \times m}$ — левые сингулярные векторы (ортонормированный базис в пространстве объектов)
- $\Sigma \in \mathbb{R}^{m \times n}$ — диагональная матрица сингулярных значений $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$
- $V \in \mathbb{R}^{n \times n}$ — правые сингулярные векторы (ортонормированный базис в пространстве признаков)

Сингулярные значения $\sigma_j$ показывают «вес» каждого направления: чем больше $\sigma_j$, тем больше информации несёт $j$-я компонента. Подробный вывод и числовые примеры — в [математическом конспекте](../../../math/lin-alg/3-alternate-coordinate-system/7-svd.md).

SVD одновременно захватывает структуру в обоих пространствах:

- $V$ — структура в пространстве **признаков** (правые сингулярные векторы = собственные векторы $X^T X$)
- $U$ — структура в пространстве **объектов** (левые сингулярные векторы = собственные векторы $X X^T$)
- $\Sigma$ — насколько сильна каждая пара $(u_j, v_j)$

Связь жёсткая: $X v_j = \sigma_j u_j$ — направление $v_j$ в пространстве признаков однозначно соответствует направлению $u_j$ в пространстве объектов с весом $\sigma_j$. Поэтому SVD работает для рекомендаций: $v_j$ — «профиль жанра» по фильмам, $u_j$ — «профиль предпочтений» по пользователям, $\sigma_j$ — насколько этот жанр значим в данных.

**Усечённый SVD (truncated SVD):** оставляем только $k$ наибольших сингулярных значений:

$$X \approx U_k \Sigma_k V_k^T$$

где $U_k \in \mathbb{R}^{m \times k}$, $\Sigma_k \in \mathbb{R}^{k \times k}$, $V_k \in \mathbb{R}^{n \times k}$. Это наилучшее приближение ранга $k$ в смысле нормы Фробениуса.

## Снижение размерности

Синтетические данные фактически двумерные ($l=100$, $n=5$, но скрытый ранг равен 2) — как если бы пять признаков порождались двумя латентными факторами плюс шум. Усечённый SVD с $k=2$ восстанавливает эту структуру.

```python
import numpy as np

np.random.seed(0)
l, n = 100, 5
# rank-2 матрица + небольшой шум: 5 признаков, но реально 2 измерения
X = np.random.randn(l, 2) @ np.random.randn(2, n) + 0.05 * np.random.randn(l, n)

# full_matrices=False — «тонкий» SVD: U (l×r), sigma (r,), Vt (r×n), r = min(l,n)
U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
print("Сингулярные значения:", sigma.round(2))
# первые два значительно больше остальных — данные ~2-мерные

k = 2
X_k = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
# относительная ошибка восстановления по норме Фробениуса
err = np.linalg.norm(X - X_k, 'fro') / np.linalg.norm(X, 'fro')
print(f"Относительная ошибка при k={k}: {err:.3f}")
```

## Регуляризация через псевдообратную

В линейной регрессии $Xw = y$ при мультиколлинеарности $X^T X$ плохо обусловлена. Через SVD решение записывается явно:

$$w^* = X^+ y = V \Sigma^+ U^T y = \sum_{j=1}^{r} \frac{1}{\sigma_j} v_j (u_j^T y)$$

Малые $\sigma_j$ дают большой вклад в $1/\sigma_j$ — решение нестабильно. Усечённый вариант (оставляем только $\sigma_j > \varepsilon$) эквивалентен регуляризации.

```python
import numpy as np


def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


np.random.seed(0)
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

K = 8  # полином степени K-1
X = np.array([[x ** i for i in range(K)] for x in coord_x])
X_train, y_train = X[::2], coord_y[::2]

U, sigma, Vt = np.linalg.svd(X_train, full_matrices=False)
print("Сингулярные значения:", sigma.round(2))

# полная псевдообратная: w* = V Σ^{-1} U^T y
w_full = Vt.T @ np.diag(1 / sigma) @ U.T @ y_train
mse_full = np.mean((X @ w_full - coord_y) ** 2)
print(f"MSE (полная псевдообратная): {mse_full:.4f}")

# усечённая: отбрасываем σ_j < ε — стабилизирует при почти-нулевых сингулярных значениях
eps = 1e-3
sigma_inv = np.where(sigma > eps, 1 / sigma, 0.0)
w_trunc = Vt.T @ np.diag(sigma_inv) @ U.T @ y_train
mse_trunc = np.mean((X @ w_trunc - coord_y) ** 2)
print(f"MSE (усечённая, eps={eps}):  {mse_trunc:.4f}")
```
