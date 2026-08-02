**Операции над векторами.** Сложение компонентное:

$$[a_1, \ldots, a_n] + [b_1, \ldots, b_n] = [a_1+b_1, \ldots, a_n+b_n].$$

**Скалярное произведение** (dot product) двух векторов $\vec{a}$ и $\vec{b}$:

$$\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n.$$

Результат — вещественное число. Основные свойства: $\vec{a}\cdot\vec{b}=\vec{b}\cdot\vec{a}$; $\lambda(\vec{a}\cdot\vec{b})=(\lambda\vec{a})\cdot\vec{b}$.

**Норма** (длина) вектора $\vec{a}$:

$$\|\vec{a}\| = \sqrt{a_1^2 + a_2^2 + \cdots + a_n^2} = \sqrt{\vec{a}\cdot\vec{a}}.$$

**Неравенство Коши–Шварца:** для любых $\vec{a}, \vec{b} \in \mathbb{R}^n$

$$|\vec{a}\cdot\vec{b}| \leq \|\vec{a}\|\,\|\vec{b}\|,$$

причём равенство достигается тогда и только тогда, когда $\vec{a}$ и $\vec{b}$ коллинеарны.

**Неравенство треугольника:**

$$\|\vec{a}+\vec{b}\| \leq \|\vec{a}\| + \|\vec{b}\|.$$

Оно вытекает из неравенства Коши–Шварца: $\|\vec{a}+\vec{b}\|^2 = \|\vec{a}\|^2 + 2\vec{a}\cdot\vec{b} + \|\vec{b}\|^2 \leq (\|\vec{a}\|+\|\vec{b}\|)^2$.
