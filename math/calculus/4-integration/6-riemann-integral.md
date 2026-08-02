**Интеграл Римана.** Пусть $f: [a, b] \to \mathbb{R}$ непрерывна. **Разбиением** $\tau$ отрезка $[a, b]$ называется набор точек $a = x_0 < x_1 < \ldots < x_n = b$. **Мелкость** разбиения $|\tau| = \max_k(x_k - x_{k-1})$. **Оснащением** $\theta$ разбиения $\tau$ называется выбор точек $\xi_k \in [x_{k-1}, x_k]$ для каждого $k$. Пара $(\tau, \theta)$ — оснащённое разбиение.

**Интегральная сумма Римана** (сумма Дарбу в общем смысле):

$$S(f, \tau, \theta) = \sum_{k=1}^n f(\xi_k)(x_k - x_{k-1}).$$

**Теорема.** Для непрерывной $f$ при $|\tau| \to 0$ интегральные суммы сходятся к $\displaystyle\int_a^b f(x)\,dx$ независимо от выбора оснащения:

$$\forall \varepsilon > 0\ \exists \delta > 0\ \forall (\tau, \theta)\colon |\tau| < \delta \;\Rightarrow\; \left|\int_a^b f(x)\,dx - S(f, \tau, \theta)\right| < \varepsilon.$$

Доказательство: по теореме Кантора (см. [равномерная непрерывность](5-uniform-continuity.md)) $f$ равномерно непрерывна на $[a, b]$: для данного $\varepsilon > 0$ найдётся $\delta > 0$ такое, что $|f(x) - f(y)| < \varepsilon/(b-a)$ при $|x - y| < \delta$. Если $|\tau| < \delta$, то

$$\int_a^b f(x)\,dx = \sum_{k=1}^n \int_{x_{k-1}}^{x_k} f(x)\,dx, \qquad S(f,\tau,\theta) = \sum_{k=1}^n f(\xi_k)(x_k - x_{k-1}) = \sum_{k=1}^n \int_{x_{k-1}}^{x_k} f(\xi_k)\,dx.$$

Разность:

$$\left|\int_a^b f - S(f,\tau,\theta)\right| = \left|\sum_{k=1}^n \int_{x_{k-1}}^{x_k}(f(x) - f(\xi_k))\,dx\right| \leq \sum_{k=1}^n \frac{\varepsilon}{b-a}(x_k - x_{k-1}) = \varepsilon.$$

**Пример 1.** $\displaystyle\lim_{n\to\infty} \sum_{k=1}^n \frac{1}{n+k}$. Функция $f(x) = \dfrac{1}{1+x}$ на $[0,1]$, разбиение $x_k = k/n$, оснащение $\xi_k = k/n$:

$$\frac{1}{n+k} = \frac{1}{n} \cdot \frac{1}{1+k/n} = f(\xi_k)(x_k - x_{k-1}) \;\Rightarrow\; \sum_{k=1}^n \frac{1}{n+k} = S(f,\tau,\theta) \to \int_0^1 \frac{dx}{1+x} = \ln(1+x)\Big|_0^1 = \ln 2.$$

**Пример 2.** $S_n(p) = 1^p + 2^p + \ldots + n^p$, $p \geq 0$. Делим на $n^{p+1}$:

$$\frac{S_n(p)}{n^{p+1}} = \sum_{k=1}^n \frac{1}{n}\left(\frac{k}{n}\right)^p = S(f,\tau,\theta) \to \int_0^1 x^p\,dx = \frac{x^{p+1}}{p+1}\bigg|_0^1 = \frac{1}{p+1}.$$

**Пример 3.** $\displaystyle S_n = n\sum_{k=1}^n \frac{1}{(n+3k)^2}$. Полагаем $x_k = k/n$, $\Delta x_k = 1/n$, $f(x) = \dfrac{1}{(1+3x)^2}$:

$$\frac{1}{(n+3k)^2} = \frac{1}{n^2(1+3k/n)^2} \;\Rightarrow\; n\sum_{k=1}^n \frac{1}{(n+3k)^2} = \sum_{k=1}^n \frac{1}{n} \cdot \frac{1}{(1+3\xi_k)^2} \to \int_0^1 \frac{dx}{(1+3x)^2} = \frac{1}{4}.$$
