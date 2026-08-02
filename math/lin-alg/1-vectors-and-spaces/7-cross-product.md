**Векторное произведение** определено только в $\mathbb{R}^3$. Для векторов $\vec{a}=(a_1,a_2,a_3)$ и $\vec{b}=(b_1,b_2,b_3)$:

$$\vec{a}\times\vec{b} = \begin{bmatrix} a_2 b_3 - a_3 b_2 \\ a_3 b_1 - a_1 b_3 \\ a_1 b_2 - a_2 b_1 \end{bmatrix}.$$

Удобный способ вычисления — через определитель:

$$\vec{a}\times\vec{b} = \begin{vmatrix} \vec{i} & \vec{j} & \vec{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{vmatrix}.$$

**Пример:** $[1,-7,1]\times[5,2,4] = [(-7)\cdot4-1\cdot2,\ 1\cdot5-1\cdot4,\ 1\cdot2-(-7)\cdot5] = [-30, 1, 37]$.

**Геометрический смысл.** Вектор $\vec{a}\times\vec{b}$ ортогонален обоим сомножителям: $(\vec{a}\times\vec{b})\cdot\vec{a}=0$ и $(\vec{a}\times\vec{b})\cdot\vec{b}=0$. Его длина равна площади параллелограмма, построенного на $\vec{a}$ и $\vec{b}$:

$$\|\vec{a}\times\vec{b}\| = \|\vec{a}\|\,\|\vec{b}\|\sin\alpha,$$

где $\alpha$ — угол между векторами.

**Тройное произведение.** Формула $\vec{a}\times(\vec{b}\times\vec{c}) = \vec{b}(\vec{a}\cdot\vec{c}) - \vec{c}(\vec{a}\cdot\vec{b})$ позволяет раскрыть двойное векторное произведение через скалярные произведения.
