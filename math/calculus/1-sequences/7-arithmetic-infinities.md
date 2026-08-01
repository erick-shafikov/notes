При вычислении пределов, содержащих бесконечности, действуют следующие правила.

**1.** Если $\lim_{n\to\infty} x_n = +\infty$ и $y_n$ ограничена снизу, то $\lim_{n\to\infty}(x_n + y_n) = +\infty$.

**2.** Если $\lim_{n\to\infty} x_n = -\infty$ и $y_n$ ограничена сверху, то $\lim_{n\to\infty}(x_n + y_n) = -\infty$.

**3.** Если $\lim_{n\to\infty} x_n = \pm\infty$ и $y_n \geq c > 0$ при всех $n$, то $\lim_{n\to\infty} x_n y_n = \pm\infty$.

**4.** Если $\lim_{n\to\infty} x_n = \pm\infty$ и $y_n \leq c < 0$ при всех $n$, то $\lim_{n\to\infty} x_n y_n = \mp\infty$.

**5.** Если $\lim_{n\to\infty} x_n = a \neq 0$, $y_n \neq 0$ и $\lim_{n\to\infty} y_n = 0$, то $\displaystyle\lim_{n\to\infty} \frac{x_n}{y_n} = \infty$.

**6.** Если $\lim_{n\to\infty} x_n = a \in \mathbb{R}$ и $\lim_{n\to\infty} y_n = \infty$, то $\displaystyle\lim_{n\to\infty} \frac{x_n}{y_n} = 0$.

**7.** Если $\lim_{n\to\infty} x_n = \infty$, $\lim_{n\to\infty} y_n = b \in \mathbb{R}$ и $y_n \neq 0$ при всех $n$, то $\displaystyle\lim_{n\to\infty} \frac{x_n}{y_n} = \infty$.

Все правила выше используют то, что одна из последовательностей не меняет знак или не обращается в ноль — именно это исключает неопределённости. Неопределённостями называют ситуации, к которым ни одно из правил не применимо напрямую:

* $(\pm\infty) + (\mp\infty)$, то есть $\infty - \infty$
* $(\pm\infty) - (\pm\infty)$
* $0 \cdot \infty$
* $\dfrac{0}{0}$
* $\dfrac{\infty}{\infty}$

**Примеры неопределённостей:**

1. $x_n = n + a$, $y_n = -n$: оба стремятся к бесконечности по модулю, сумма $x_n + y_n = a$ конечна — тип $\infty - \infty$.

2. $x_n = n^2 + n$, $y_n = n^2$: оба стремятся к $+\infty$, разность $x_n - y_n = n \to +\infty$ — тип $\infty - \infty$, но результат сам бесконечен.
