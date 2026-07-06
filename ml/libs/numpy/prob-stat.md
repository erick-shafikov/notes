# numpy — вероятность и статистика

## np.mean

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
