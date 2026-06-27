# Равномерное распределение

НСВ $X$ имеет **равномерное распределение** на $[a, b]$, если [плотность распределения](4-probability-density.md) имеет вид:

$$f(x) = \begin{cases} C, & a \leq x \leq b \\ 0, & \text{иначе} \end{cases}$$

Из условия нормировки $\displaystyle\int_{-\infty}^{+\infty} f(x)\,dx = 1$:

$$C \cdot (b - a) = 1 \implies C = \frac{1}{b - a}$$

Таким образом, плотность равномерного распределения:

$$f(x) = \begin{cases} 0, & x < a \\ \dfrac{1}{b-a}, & a \leq x \leq b \\ 0, & x > b \end{cases}$$

## Функция распределения

$$F(x) = \int_{-\infty}^{x} f(t)\,dt$$

Вычисляем по случаям:

- $x < a$: $F(x) = \displaystyle\int_{-\infty}^{x} 0\,dt = 0$

- $a \leq x \leq b$: $F(x) = \displaystyle\int_{-\infty}^{a} 0\,dt + \int_{a}^{x} \frac{dt}{b-a} = \frac{x - a}{b - a}$

- $x > b$: $F(x) = \displaystyle\int_{-\infty}^{a} 0\,dt + \int_{a}^{b} \frac{dt}{b-a} + \int_{b}^{x} 0\,dt = 1$

$$F(x) = \begin{cases} 0, & x < a \\ \dfrac{x - a}{b - a}, & a \leq x \leq b \\ 1, & x > b \end{cases}$$

График $F(x)$: линейно растёт от 0 до 1 на отрезке $[a, b]$.

## Математическое ожидание

$$M(X) = \int_{-\infty}^{+\infty} x\,f(x)\,dx = \int_{a}^{b} \frac{x}{b-a}\,dx = \frac{1}{b-a} \cdot \frac{x^2}{2}\bigg|_a^b = \frac{a + b}{2}$$

## Дисперсия

$$D(X) = \int_{a}^{b} x^2\,f(x)\,dx - \bigl[M(X)\bigr]^2 = \frac{1}{b-a}\int_a^b x^2\,dx - \frac{(a+b)^2}{4} = \frac{(b-a)^2}{12}$$

## Вероятность попадания в подынтервал

$$P(\alpha < X < \beta) = \int_{\alpha}^{\beta} \frac{dx}{b-a} = \frac{\beta - \alpha}{b - a}$$

то есть вероятность пропорциональна длине подынтервала.

## Примеры

_Пример 1._ $X \in [2, 6]$.

$$M(X) = \frac{2 + 6}{2} = 4, \qquad D(X) = \frac{(6-2)^2}{12} = \frac{16}{12} = \frac{4}{3}$$

_Пример 2._ Автобусы ходят каждые 10 минут. Пассажир пришёл в случайный момент после отправления автобуса, т.е. $X \sim U(0, 10)$. Найти вероятность того, что он прождёт более 2 минут.

Если пассажир пришёл в момент $X$ (секунд после отправления), то ждать осталось $10 - X$ минут. Условие «ждать более 2 минут»: $10 - X > 2$, то есть $X \in (0, 8)$.

$$P(0 < X < 8) = \frac{8 - 0}{10 - 0} = 0{,}8$$
