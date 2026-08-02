**Проекция на прямую.** Пусть $L = \{c\vec{v} \mid c \in \mathbb{R}\}$ — прямая в $\mathbb{R}^n$, задаваемая вектором $\vec{v}$. Проекция вектора $\vec{x}$ на $L$ — это такой вектор $\operatorname{Proj}(\vec{x}) = c\vec{v}$, что разность $\vec{x} - c\vec{v}$ ортогональна $L$:

$$(\vec{x} - c\vec{v}) \cdot \vec{v} = 0 \;\Longrightarrow\; c = \frac{\vec{x} \cdot \vec{v}}{\vec{v} \cdot \vec{v}}.$$

Поэтому:

$$\operatorname{Proj}_L(\vec{x}) = \frac{\vec{x} \cdot \vec{v}}{\vec{v} \cdot \vec{v}}\,\vec{v}.$$

**Пример.** $L = \{c(2,1)\}$, $\vec{x} = (2,3)$:

$$c = \frac{(2,3)\cdot(2,1)}{(2,1)\cdot(2,1)} = \frac{7}{5}, \quad \operatorname{Proj}_L(\vec{x}) = \frac{7}{5}\begin{pmatrix}2\\1\end{pmatrix} = \begin{pmatrix}14/5\\7/5\end{pmatrix}.$$

**Проекция как линейное преобразование.** Оператор $\operatorname{Proj}_L: \mathbb{R}^n \to \mathbb{R}^n$ линеен:

$$\operatorname{Proj}_L(\vec{x}+\vec{y}) = \frac{(\vec{x}+\vec{y})\cdot\vec{v}}{\vec{v}\cdot\vec{v}}\,\vec{v} = \operatorname{Proj}_L(\vec{x}) + \operatorname{Proj}_L(\vec{y}),$$

и аналогично для умножения на скаляр. Если $\hat{m} = \vec{v}/|\vec{v}|$ — единичный вектор вдоль $L$, то $\operatorname{Proj}_L(\vec{x}) = (\vec{x}\cdot\hat{m})\hat{m}$.

**Матрица проекции.** Для $\hat{m} = (m_1, m_2)$ в $\mathbb{R}^2$ матрица имеет вид:

$$A = \begin{bmatrix}m_1^2 & m_1 m_2 \\ m_1 m_2 & m_2^2\end{bmatrix}.$$

**Пример.** $\vec{v} = (2,1)$, $|\vec{v}| = \sqrt{5}$, $\hat{m} = (2/\sqrt{5},\, 1/\sqrt{5})$:

$$A = \begin{bmatrix}4/5 & 2/5 \\ 2/5 & 1/5\end{bmatrix}, \qquad \operatorname{Proj}_L(\vec{x}) = A\vec{x}.$$
