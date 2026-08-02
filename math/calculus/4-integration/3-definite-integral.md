**Площадь и мера.** Пусть $\mathcal{F}$ — семейство всех ограниченных подмножеств плоскости, представимых в виде прямоугольников $(a_1, b_1) \times (a_2, b_2)$. Площадь прямоугольника $S = (b_1 - a_1)(b_2 - a_2)$. Функция $S: \mathcal{F} \to [0, +\infty)$ называется площадью, если:

1. $S((a_1, b_1) \times (a_2, b_2)) = (b_1 - a_1)(b_2 - a_2)$,
2. $S(E) = S(E_1) + S(E_2)$, если $E = E_1 \cup E_2$ и $E_1 \cap E_2 = \varnothing$,
3. $\widetilde{E} \subseteq E \Rightarrow S(\widetilde{E}) \leq S(E)$.

Для произвольного $E \subseteq \mathbb{R}^2$ определяют **внешние меры**: $\sigma_1(E) = \inf\!\left\{\sum \sigma(P_i) : \bigcup_{i=1}^{n} P_i \supseteq E\right\}$ (минимум по конечным покрытиям прямоугольниками) и $\sigma_2(E) = \inf\!\left\{\sum_{i=0}^{\infty} \sigma(P_i) : \bigcup_{i=1}^{\infty} P_i \supseteq E\right\}$ (по счётным). Всегда $\sigma_1(E) \geq \sigma_2(E)$. Например, для $E = ([0,1] \cap \mathbb{Q}) \times ([0,1] \setminus \mathbb{Q})$: $\sigma_1(E) = 1$, но $\sigma_2(E) = 0$, поскольку $E$ покрывается счётным семейством прямоугольников нулевой суммарной площади.

**Положительная и отрицательная части.** Для функции $f: E \to \mathbb{R}$ определяют

$$f_+(x) = \max\{f(x), 0\}, \qquad f_-(x) = \max\{-f(x), 0\}.$$

Обе части неотрицательны, $f = f_+ - f_-$ и $|f| = f_+ + f_-$, причём

$$f_+ = \frac{f + |f|}{2}, \qquad f_- = \frac{|f| - f}{2}.$$

Если $f$ непрерывна, то $f_+$ и $f_-$ тоже непрерывны.

**Подграфик** функции $f: [a,b] \to [0, +\infty)$ — это множество $P_f = \{(x, y) \in \mathbb{R}^2 : x \in [a, b],\; 0 \leq y < f(x)\}$.

**Определение интеграла.** Для непрерывной $f: [a, b] \to \mathbb{R}$ полагают

$$\int_a^b f = \sigma(P_{f_+}) - \sigma(P_{f_-}).$$

где $\sigma$ — площадь подграфиков.

**Свойства определённого интеграла.** Пусть $f, g: [a, b] \to \mathbb{R}$ непрерывны.

1. $\displaystyle\int_a^a f = 0$.
2. $f \geq 0$ на $[a, b]$ влечёт $\displaystyle\int_a^b f \geq 0$.
3. $\displaystyle\int_a^b (-f) = -\int_a^b f$, поскольку $(-f)_+ = f_-$ и $(-f)_- = f_+$.
4. $\displaystyle\int_a^b c\,dx = c(b - a)$ для любой константы $c$.
5. Если $f \geq 0$ и $\displaystyle\int_a^b f = 0$, то $f \equiv 0$ на $[a, b]$.

**Аддитивность по промежутку.** Для любого $c \in [a, b]$:

$$\int_a^b f = \int_a^c f + \int_c^b f.$$

Это позволяет определить $\displaystyle\int_a^b f = -\int_b^a f$ при $a > b$ и распространить аддитивность на произвольный порядок концов. Для разбиения $a \leq c_1 \leq c_2 \leq \ldots \leq c_n \leq b$:

$$\int_a^b f = \int_a^{c_1} f + \int_{c_1}^{c_2} f + \ldots + \int_{c_n}^b f.$$

**Неравенство интеграла.** Если $f \leq g$ на $[a, b]$, то $\displaystyle\int_a^b f \leq \int_a^b g$. Следствие:

$$(b - a)\min_{x \in [a,b]} f(x) \leq \int_a^b f \leq (b - a)\max_{x \in [a,b]} f(x).$$

Из неравенства $-|f| \leq f \leq |f|$ получают $\displaystyle\left|\int_a^b f\right| \leq \int_a^b |f|$.

**Теорема о среднем значении.** Если $f: [a, b] \to \mathbb{R}$ непрерывна, то существует $c \in [a, b]$ такое, что

$$\int_a^b f = (b - a)\,f(c).$$

Доказательство: из оценки $(b-a)\min f \leq \int_a^b f \leq (b-a)\max f$ следует, что $\dfrac{1}{b-a}\displaystyle\int_a^b f \in [\min f, \max f] = f([a,b])$ (по теореме Больцано-Коши о промежуточных значениях для непрерывной $f$), откуда существует нужное $c$.
