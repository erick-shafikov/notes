**Замена переменной (подстановка).** Пусть $\varphi: [\alpha, \beta] \to \mathbb{R}$ дифференцируема и $F$ — первообразная $f$. Тогда

$$\int f(\varphi(x))\,\varphi'(x)\,dx = F(\varphi(x)) + C.$$

Записывают это как замену $y = \varphi(x)$, $dy = \varphi'(x)\,dx$:

$$\int f(\varphi(x))\,\varphi'(x)\,dx = \int f(y)\,dy = F(y) + C = F(\varphi(x)) + C.$$

Пример 1: $\displaystyle\int \frac{x}{x^2+1}\,dx = \frac{1}{2}\int \frac{(x^2+1)'}{x^2+1}\,dx$. Полагая $y = x^2+1$, $dy = 2x\,dx$:

$$= \frac{1}{2}\int \frac{dy}{y} = \frac{1}{2}\ln|y| + C = \frac{1}{2}\ln(x^2+1) + C.$$

Пример 2: $\displaystyle\int \frac{dx}{1 + \sqrt[3]{x}}$. Полагаем $t = \sqrt[3]{x}$, то есть $x = t^3$, $dx = 3t^2\,dt$:

$$= \int \frac{3t^2\,dt}{1+t} = 3\int \frac{t^2}{1+t}\,dt = 3\int \frac{t^2 - 1 + 1}{t+1}\,dt = 3\int \!\left(t - 1 + \frac{1}{t+1}\right)\!dt = \frac{3t^2}{2} - 3t + 3\ln|t+1| + C.$$

где в последнем шаге использовано тождество $\dfrac{t^2}{t+1} = t - 1 + \dfrac{1}{t+1}$.

---

**Интегрирование по частям.** Из правила Лейбница $(fg)' = f'g + fg'$ следует

$$\int f(x)\,g'(x)\,dx = f(x)\,g(x) - \int f'(x)\,g(x)\,dx.$$

В сокращённой форме, полагая $u = f(x)$, $dv = g'(x)\,dx$, $v = g(x)$, $du = f'(x)\,dx$:

$$\int u\,dv = uv - \int v\,du.$$

Пример 1: $\displaystyle\int \ln x\,dx$. Берём $u = \ln x$, $dv = dx$, тогда $v = x$ и $du = \dfrac{dx}{x}$:

$$\int \ln x\,dx = x\ln x - \int x \cdot \frac{dx}{x} = x\ln x - \int dx = x\ln x - x + C.$$

Пример 2: $\displaystyle\int x^2 e^x\,dx$. Берём $u = x^2$, $dv = e^x\,dx$, тогда $v = e^x$, $du = 2x\,dx$:

$$\int x^2 e^x\,dx = x^2 e^x - \int 2x e^x\,dx.$$

К оставшемуся интегралу применяем ещё раз: $u = x$, $dv = e^x\,dx$, $v = e^x$, $du = dx$:

$$\int 2x e^x\,dx = 2\!\left(x e^x - \int e^x\,dx\right) = 2xe^x - 2e^x + C.$$

Итого: $\displaystyle\int x^2 e^x\,dx = x^2 e^x - 2xe^x + 2e^x + C$.
