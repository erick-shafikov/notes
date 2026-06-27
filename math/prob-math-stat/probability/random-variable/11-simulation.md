# Моделирование случайных величин (обратный метод)

## Псевдослучайные числа и базовый генератор

Базовый генератор `X = rand()` даёт псевдослучайные числа с **равномерным распределением** на $[0, 1]$:

$$\omega(x) = \begin{cases} \dfrac{1}{b-a}, & a \leq x \leq b \\ 0, & \text{иначе} \end{cases} \quad \text{при } a=0,\; b=1$$

Числовые характеристики [равномерного распределения](7-uniform-distribution.md):
$$m_x = \frac{a+b}{2}, \qquad \sigma_x^2 = \frac{(b-a)^2}{12}$$

## Моделирование равномерной СВ на $[a, b]$

Из системы:
$$\begin{cases} \dfrac{a+b}{2} = m_Y \\[6pt] \dfrac{(b-a)^2}{12} = \sigma_Y^2 \end{cases}$$

Из первого уравнения $a = 2m_Y - b$; подставляем во второе:

$$\frac{(2b - 2m_Y)^2}{12} = \sigma_Y^2 \implies 4(b-m_Y)^2 = 12\sigma_Y^2 \implies b - m_Y = \pm\sqrt{3}\,\sigma_Y$$

Квадратное уравнение $b^2 - 2bm_Y + m_Y^2 - 3\sigma_Y^2 = 0$ даёт $D = 12\sigma_Y^2$:

$$b_{1,2} = m_Y \pm \sqrt{3}\,\sigma_Y$$

Берём: $b = m_Y + \sqrt{3}\,\sigma_Y$, $\;a = m_Y - \sqrt{3}\,\sigma_Y$.

**Формула преобразования:**

$$Y = a + X\cdot(b - a), \quad X = \texttt{rand()} \sim U(0,1)$$

даёт $Y \sim U(a, b)$ с заданными $m_Y$ и $\sigma_Y$.

## Моделирование нормальной СВ (метод ЦПТ)

$Y$ — нормальная СВ с параметрами $m_Y$, $\sigma_Y^2$:

$$\omega(y) = \frac{1}{\sigma_Y\sqrt{2\pi}}\, e^{-\frac{(y - m_Y)^2}{2\sigma_Y^2}}$$

По центральной предельной теореме сумма $N$ независимых одинаково распределённых СВ:

$$L = \frac{1}{N}\sum_{i=1}^N x_i, \quad x_i \sim U(0,1)$$

имеет параметры:

$$m_Y = \frac{1}{N}\sum_{i=1}^N M\{x_i\} = m_x, \qquad \sigma_Y^2 = \frac{\sigma_x^2}{N} \xrightarrow{N\to\infty} \frac{\sigma_x^2}{N}$$

При $N \to \infty$ распределение $L$ стремится к нормальному. Практически достаточно $N = 12$.

## Моделирование распределения Рэлея

$$\omega(z) = \frac{z}{\sigma^2}\,e^{-z^2/(2\sigma^2)}, \quad z \geq 0$$

Если $Y_1, Y_2$ — нормальные независимые СВ с $m_Y = 0$ и дисперсией $\sigma_Y^2$, то:

$$Z = \sqrt{Y_1^2 + Y_2^2}$$

имеет распределение Рэлея с параметрами:

$$m_Z = \sqrt{\frac{\pi}{2}}\,\sigma_Y, \qquad \sigma_Z^2 = \left(2 - \frac{\pi}{2}\right)\sigma_Y^2$$

## Моделирование показательной СВ (обратный метод)

[Показательное распределение](8-exponential-distribution.md): $\omega(y) = \lambda e^{-\lambda y}$, $m_Y = 1/\lambda$, $\sigma_Y^2 = 1/\lambda^2$.

Функция распределения $F(y) = 1 - e^{-\lambda y}$. Из $F(Y) = X$ при $X \sim U(0,1)$:

$$1 - e^{-\lambda Y} = X \implies e^{-\lambda Y} = 1 - X \implies Y = -\frac{1}{\lambda}\ln(1-X)$$

Так как $1 - X$ тоже равномерно на $[0,1]$:

$$\boxed{Y = -\frac{1}{\lambda}\ln X, \quad X = \texttt{rand()}}$$
