# Вероятность отклонения относительной частоты от вероятности

Если в схеме $n$ независимых испытаний событие $A$ наступает с вероятностью $p$, а $m$ — число его появлений, то $m/n$ — **относительная частота**. Для любого $\varepsilon > 0$:

$$P\!\left(\left|\frac{m}{n} - p\right| < \varepsilon\right) = 2\Phi\!\left(\varepsilon\sqrt{\frac{n}{pq}}\right), \qquad q = 1-p$$

где $\Phi(x)$ — интегральная функция Лапласа из [теоремы Лапласа](3-integral-laplace-theorem.md).

**Вывод.** Неравенство $\left|\frac{m}{n} - p\right| < \varepsilon$ равносильно $np - \varepsilon n < m < np + \varepsilon n$. По интегральной теореме Лапласа:

$$P(np - \varepsilon n < m < np + \varepsilon n) = \Phi\!\left(\frac{n\varepsilon}{\sqrt{npq}}\right) - \Phi\!\left(\frac{-n\varepsilon}{\sqrt{npq}}\right) = 2\Phi\!\left(\varepsilon\sqrt{\frac{n}{pq}}\right)$$

При $n \to \infty$ аргумент $\Phi$ стремится к $+\infty$, значит $\Phi \to \frac{1}{2}$ и вся вероятность стремится к 1:

$$\lim_{n \to \infty} P\!\left(\left|\frac{m}{n} - p\right| < \varepsilon\right) = 1$$

Это означает: при достаточно большом числе испытаний относительная частота сколь угодно мало отклоняется от вероятности — **закон больших чисел Бернулли**.

_Пример 1._ Проводится $n = 625$ испытаний, $p = 0{,}8$. Найти вероятность того, что относительная частота отклонится от $p$ не более чем на $\varepsilon = 0{,}04$.

$$P\!\left(\left|\frac{m}{625} - 0{,}8\right| < 0{,}04\right) = 2\Phi\!\left(0{,}04\sqrt{\frac{625}{0{,}8 \cdot 0{,}2}}\right) = 2\Phi\!\left(0{,}04 \cdot 62{,}5\right) = 2\Phi(2{,}5) \approx 0{,}9876$$

_Пример 2 (обратная задача)._ Вероятность $p = 0{,}5$. Сколько нужно испытаний $n$, чтобы с вероятностью $P = 0{,}7698$ относительная частота отклонялась от $p$ не более чем на $\varepsilon = 0{,}02$?

Из условия $2\Phi\!\left(\varepsilon\sqrt{n/pq}\right) = 0{,}7698$ следует $\Phi(\ldots) = 0{,}3849$. По таблице $\Phi(1{,}2) = 0{,}3849$, значит:

$$0{,}02\sqrt{\frac{n}{0{,}5 \cdot 0{,}5}} = 1{,}2 \implies 0{,}02 \cdot 2\sqrt{n} = 1{,}2 \implies \sqrt{n} = 30 \implies n = 900$$
