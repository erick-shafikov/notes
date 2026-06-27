# Плотность распределения непрерывной случайной величины

Пусть $X$ — непрерывная СВ с [функцией распределения](3-continuous-random-variable.md) $F(x)$. Вероятность попадания в малый интервал $(x;\, x+\Delta x)$:

$$P(x < X < x+\Delta x) = F(x+\Delta x) - F(x) \approx F'(x)\cdot\Delta x$$

При $\Delta x \to 0$ отношение $\dfrac{F(x+\Delta x)-F(x)}{\Delta x} \to F'(x)$. Это предел называется **плотностью распределения**:

$$f(x) = F'(x)$$

Геометрически $f(x)$ — кривая распределения; площадь под ней на промежутке $[a,b]$ равна вероятности попадания туда.

**Свойства плотности:**

1. $f(x) \geq 0$ для всех $x \in (-\infty,+\infty)$
2. $\displaystyle P(a < X < b) = \int_a^b f(x)\,dx$
3. $\displaystyle F(x) = \int_{-\infty}^{x} f(t)\,dt \quad$ (т.е. $P(X < x)$)
4. $\displaystyle\int_{-\infty}^{+\infty} f(x)\,dx = 1$ (условие нормировки)

_Пример 1: равномерное распределение._ Грань кубика выпадает равновероятно: $p_i = 1/6$. В непрерывном аналоге (точка равномерно на $[0,6]$):

$$f(x) = \begin{cases} \dfrac{1}{6}, & x \in [0,6] \\ 0, & \text{иначе} \end{cases} \implies F(x) = \frac{x}{6}$$

$$P(3 < X < 4) = F(4) - F(3) = \frac{4}{6} - \frac{3}{6} = \frac{1}{6}$$

_Пример 2: найти $C$ в плотности $f(x) = \dfrac{C}{1+x^2}$._ По свойству 4:

$$\int_{-\infty}^{+\infty} \frac{C}{1+x^2}\,dx = C\cdot\arctan(x)\Big|_{-\infty}^{+\infty} = C\!\left(\frac{\pi}{2} - \left(-\frac{\pi}{2}\right)\right) = C\pi = 1 \implies C = \frac{1}{\pi}$$

_Пример 3: найти $a$ и $P(2 < X < 4)$ для_

$$f(x) = \begin{cases} 0, & x < 1 \\ ax, & 1 \leq x \leq 3 \\ 0, & x > 3 \end{cases}$$

По нормировке:

$$\int_1^3 ax\,dx = a\cdot\frac{x^2}{2}\bigg|_1^3 = a\cdot\frac{9-1}{2} = 4a = 1 \implies a = \frac{1}{4}$$

Вычислим $P(2 < X < 4)$. Так как $f(x) = 0$ при $x > 3$, верхний предел интегрирования фактически равен 3:

$$P(2 < X < 4) = \int_2^3 \frac{x}{4}\,dx + \int_3^4 0\,dx = \frac{1}{4}\cdot\frac{x^2}{2}\bigg|_2^3 = \frac{1}{8}(9-4) = \frac{5}{8}$$
