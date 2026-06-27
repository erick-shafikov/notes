# Показательное (экспоненциальное) распределение

СВ $X$ имеет **показательное распределение** с параметром $\lambda > 0$, если [плотность распределения](4-probability-density.md) имеет вид:

$$f(x) = \begin{cases} 0, & x < 0 \\ \lambda e^{-\lambda x}, & x \geq 0 \end{cases}$$

## Функция распределения

$$F(x) = \int_{-\infty}^{x} f(t)\,dt = \int_{-\infty}^{0} 0\,dt + \int_{0}^{x} \lambda e^{-\lambda t}\,dt = \lambda \cdot \left[-\frac{e^{-\lambda t}}{\lambda}\right]_0^x = 1 - e^{-\lambda x}$$

$$F(x) = \begin{cases} 0, & x < 0 \\ 1 - e^{-\lambda x}, & x \geq 0 \end{cases}$$

График $f(x)$: убывающая экспонента при $x \geq 0$. График $F(x)$: вогнутая кривая, возрастающая от 0 к 1.

## Математическое ожидание

$$M(X) = \int_{0}^{+\infty} x\,\lambda e^{-\lambda x}\,dx = \frac{1}{\lambda}$$

## Дисперсия

$$D(X) = \int_{0}^{+\infty} x^2\,\lambda e^{-\lambda x}\,dx - \bigl[M(X)\bigr]^2 = \frac{1}{\lambda^2}$$

Таким образом, $\sigma(X) = \dfrac{1}{\lambda} = M(X)$ — характерное свойство показательного распределения.

## Вероятность попадания в интервал

Для $0 \leq \alpha < \beta$:

$$P(\alpha < X < \beta) = F(\beta) - F(\alpha) = e^{-\lambda\alpha} - e^{-\lambda\beta}$$

## Пример

Среднее время расчёта — 10 минут, т.е. $M(X) = 1/\lambda = 10$, откуда $\lambda = 0{,}1$.

Найти: а) $P(X < 5)$ — вероятность завершить за 5 минут; б) $P(5 < X < 15)$.

**а)**

$$P(0 < X < 5) = 1 - e^{-0{,}1 \cdot 5} = 1 - e^{-0{,}5} \approx 1 - 0{,}606 = 0{,}394$$

**б)**

$$P(5 < X < 15) = e^{-0{,}1 \cdot 5} - e^{-0{,}1 \cdot 15} = e^{-0{,}5} - e^{-1{,}5} \approx 0{,}606 - 0{,}224 = 0{,}382$$
