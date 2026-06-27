# Неравенство и теорема Чебышёва

## Неравенство Чебышёва

Пусть СВ $X$ имеет математическое ожидание $m_x$ и дисперсию $\sigma_x^2$. Тогда для любого $\alpha > 0$:

$$P(|X - m_x| \geq \alpha) \leq \frac{\sigma_x^2}{\alpha^2}$$

**Следствие:**

$$P(|X - m_x| < \alpha) \geq 1 - \frac{\sigma_x^2}{\alpha^2}$$

## Доказательство (дискретный случай)

По определению дисперсии:

$$\sigma_x^2 = \sum_{i=1}^n (x_i - m_x)^2\, p_i$$

Разобьём сумму на два слагаемых: по $i$, где $|x_i - m_x| < \alpha$, и по $i$, где $|x_i - m_x| \geq \alpha$:

$$\sigma_x^2 \geq \sum_{|x_i - m_x|\,\geq\,\alpha} (x_i - m_x)^2\, p_i \geq \alpha^2 \sum_{|x_i - m_x|\,\geq\,\alpha} p_i = \alpha^2 \cdot P(|X - m_x| \geq \alpha)$$

Делим на $\alpha^2$: $\quad P(|X - m_x| \geq \alpha) \leq \dfrac{\sigma_x^2}{\alpha^2} \quad \blacksquare$

## Пример

При $\alpha = 3\sigma_x$:

$$P(|X - m_x| \geq 3\sigma_x) \leq \frac{\sigma_x^2}{9\sigma_x^2} = \frac{1}{9} \approx 0{,}111$$

Неравенство Чебышёва даёт грубую оценку, не зависящую от вида распределения (для нормального закона эта вероятность ≈ 0.0027, то есть правило $3\sigma$ значительно точнее).

## Теорема Чебышёва (Закон Больших Чисел)

Пусть $X_1, X_2, \ldots, X_n$ — независимые СВ с одинаковыми $m_x$ и $\sigma_x^2$. Рассмотрим выборочное среднее:

$$Y = \frac{1}{n}\sum_{i=1}^n X_i, \qquad m_Y = m_x, \quad \sigma_Y^2 = \frac{\sigma_x^2}{n}$$

Применяем неравенство Чебышёва к $Y$:

$$P(|Y - m_x| \geq \varepsilon) \leq \frac{\sigma_Y^2}{\varepsilon^2} = \frac{\sigma_x^2}{n\varepsilon^2}$$

При $n \to \infty$ правая часть стремится к 0, поэтому:

$$P\!\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - m_x\right| < \varepsilon\right) \geq 1 - \frac{\sigma_x^2}{n\varepsilon^2} \xrightarrow{n\to\infty} 1$$

**Смысл:** при достаточно большом числе наблюдений выборочное среднее с высокой вероятностью сколь угодно близко к теоретическому $m_x$.
