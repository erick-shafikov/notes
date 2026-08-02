**Первый замечательный предел:**

$$\lim_{x \to 0} \frac{\sin x}{x} = 1.$$

*Доказательство.* При $0 < x < \dfrac{\pi}{2}$ из сравнения площадей (треугольник $OBA$ < сектор $OCA$ < треугольник $OCA$ единичной окружности) получаем $\sin x < x < \tan x$. Деля на $\sin x > 0$:

$$1 < \frac{x}{\sin x} < \frac{1}{\cos x}, \quad \text{то есть} \quad \cos x < \frac{\sin x}{x} < 1.$$

Так как $\cos x \to 1$ при $x \to 0$, по теореме о сжатой функции $\dfrac{\sin x}{x} \to 1$. Для $x < 0$ используют нечётность синуса.

Следствия первого замечательного предела:

$$\lim_{x \to 0} \frac{1 - \cos x}{x^2} = \frac{1}{2}, \qquad \text{поскольку} \quad \frac{1-\cos x}{x^2} = \frac{\sin^2 x}{x^2(1 + \cos x)} \to 1 \cdot \frac{1}{2}.$$

$$\lim_{x \to 0} \frac{\arcsin x}{x} = 1, \qquad \text{замена } x = \sin t,\ t \to 0\colon \quad \frac{\arcsin x}{x} = \frac{t}{\sin t} \to 1.$$

$$\lim_{x \to 0} \frac{\arctan x}{x} = 1, \qquad \text{замена } x = \tan t,\ t \to 0\colon \quad \frac{\arctan x}{x} = \frac{t}{\tan t} \to 1.$$

---

**Второй замечательный предел:**

$$\lim_{x \to 0} (1 + x)^{1/x} = e, \qquad \text{эквивалентно} \quad \lim_{n \to \infty}\Bigl(1 + \tfrac{1}{n}\Bigr)^n = e.$$

где $e$ — основание натурального логарифма, $e \approx 2{,}718\ldots$

Следствия второго замечательного предела:

$$\lim_{x \to 0} \frac{\ln(1 + x)}{x} = 1, \qquad \text{поскольку} \quad \frac{\ln(1+x)}{x} = \ln(1+x)^{1/x} \to \ln e = 1.$$

$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1, \qquad \text{замена } t = e^x - 1,\ x = \ln(1+t),\ t\to 0.$$

$$\lim_{x \to 0} \frac{a^x - 1}{x} = \ln a \quad (a > 0),\qquad \text{поскольку} \quad \frac{a^x-1}{x} = \frac{e^{x\ln a}-1}{x\ln a}\cdot\ln a \to \ln a.$$

$$\lim_{x \to 0} \frac{(1+x)^n - 1}{x} = n \quad (n \in \mathbb{R}),\qquad \text{поскольку} \quad (1+x)^n - 1 = e^{n\ln(1+x)}-1 \sim n\ln(1+x) \sim nx.$$

**Пример вычисления с помощью эквивалентностей.** Пусть требуется найти $\displaystyle\lim_{x\to 0}\frac{\ln\cos x}{x^2}$. Поскольку $\cos x - 1 \sim -\dfrac{x^2}{2}$ и $\ln(1+u) \sim u$ при $u \to 0$:

$$\frac{\ln\cos x}{x^2} = \frac{\ln(1+(\cos x - 1))}{x^2} \sim \frac{\cos x - 1}{x^2} \sim \frac{-x^2/2}{x^2} = -\frac{1}{2}.$$
