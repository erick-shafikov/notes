**Проекция на подпространство.** Пусть $V \subseteq \mathbb{R}^n$ — подпространство и $A = \begin{bmatrix}\vec{v}_1 & \cdots & \vec{v}_k\end{bmatrix}$ — матрица, столбцы которой образуют базис $V$. Для любого $\vec{x} \in \mathbb{R}^n$ его проекция на $V$ определяется как ближайшая к $\vec{x}$ точка в $V$:

$$\operatorname{Proj}_V \vec{x} = A(A^T A)^{-1} A^T \vec{x}.$$

Матрица $P = A(A^T A)^{-1} A^T$ называется матрицей проекции на $V$. Разложение $\vec{x} = \operatorname{Proj}_V \vec{x} + \vec{w}$, где $\vec{w} = \vec{x} - \operatorname{Proj}_V \vec{x} \in V^\perp$, единственно.

Вывод формулы: вектор $\vec{a} = A\vec{y} \in V$ является проекцией тогда и только тогда, когда $\vec{x} - A\vec{y} \perp V$, то есть $A^T(\vec{x} - A\vec{y}) = \vec{0}$. Отсюда $A^T A \vec{y} = A^T \vec{x}$, и при линейно независимых столбцах $A$ матрица $A^T A$ обратима, что даёт $\vec{y} = (A^T A)^{-1} A^T \vec{x}$.

**Пример.** $V = \operatorname{span}\!\left(\begin{bmatrix}1\\0\\0\\1\end{bmatrix},\begin{bmatrix}0\\1\\0\\1\end{bmatrix}\right) \subseteq \mathbb{R}^4$. Тогда $A = \begin{bmatrix}1&0\\0&1\\0&0\\1&1\end{bmatrix}$, $A^T A = \begin{bmatrix}2&1\\1&2\end{bmatrix}$, $(A^T A)^{-1} = \tfrac{1}{3}\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$.

**Проекция на $N(D)$.** Если подпространство задано как $V = N(D)$, то $V^\perp = C(D^T)$ и

$$\operatorname{Proj}_{V^\perp} \vec{x} = D^T (D D^T)^{-1} D \vec{x}, \qquad \operatorname{Proj}_V \vec{x} = \vec{x} - \operatorname{Proj}_{V^\perp} \vec{x}.$$

Например, $V = \{(x_1, x_2, x_3) \mid x_1 + x_2 + x_3 = 0\} = N([1,1,1])$. Тогда $\operatorname{Proj}_{V^\perp} \vec{x} = \tfrac{1}{3}\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}\vec{x}$ и $\operatorname{Proj}_V \vec{x} = \tfrac{1}{3}\begin{bmatrix}2&-1&-1\\-1&2&-1\\-1&-1&2\end{bmatrix}\vec{x}$.

**Метод наименьших квадратов (МНК).** Когда система $A\vec{x} = \vec{b}$ несовместна, ищем $\vec{x}^*$, минимизирующее $\|\vec{b} - A\vec{x}\|^2$. Геометрически это означает, что $A\vec{x}^* = \operatorname{Proj}_{C(A)} \vec{b}$. Из условия $A^T(\vec{b} - A\vec{x}^*) = \vec{0}$ получаем **нормальные уравнения**:

$$A^T A \vec{x}^* = A^T \vec{b}.$$

**Пример: линейная аппроксимация.** По точкам $(-1,0)$, $(0,1)$, $(1,2)$, $(2,1)$ ищем прямую $y = mx + b$. Система $A\vec{z} = \vec{b}$ с $A = \begin{bmatrix}-1&1\\0&1\\1&1\\2&1\end{bmatrix}$, $\vec{b} = \begin{bmatrix}0\\1\\2\\1\end{bmatrix}$ несовместна. Нормальные уравнения:

$$A^T A = \begin{bmatrix}6&2\\2&4\end{bmatrix}, \quad A^T \vec{b} = \begin{bmatrix}4\\4\end{bmatrix}.$$

Решая, получаем $m^* = \tfrac{2}{5}$, $b^* = \tfrac{4}{5}$, то есть наилучшая прямая $y = \tfrac{2}{5}x + \tfrac{4}{5}$.
