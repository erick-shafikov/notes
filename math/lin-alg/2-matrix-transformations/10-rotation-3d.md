**Поворот в $\mathbb{R}^3$** вокруг оси $x$ на угол $\alpha$ — линейное преобразование $\operatorname{Rot}_\alpha^{(3)}: \mathbb{R}^3 \to \mathbb{R}^3$.

Стандартный базис $\{e_1, e_2, e_3\}$ при повороте вокруг оси $x$: вектор $e_1 = (1,0,0)$ лежит на оси вращения и не изменяется; векторы $e_2$ и $e_3$ вращаются в плоскости $yz$ так же, как при 2D-повороте. Поэтому:

$$\operatorname{Rot}_\alpha^{(3)}(e_1) = \begin{pmatrix}1\\0\\0\end{pmatrix}, \quad \operatorname{Rot}_\alpha^{(3)}(e_2) = \begin{pmatrix}0\\\cos\alpha\\\sin\alpha\end{pmatrix}, \quad \operatorname{Rot}_\alpha^{(3)}(e_3) = \begin{pmatrix}0\\-\sin\alpha\\\cos\alpha\end{pmatrix}.$$

Матрица поворота вокруг оси $x$:

$$A = \begin{bmatrix}1 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha \\ 0 & \sin\alpha & \cos\alpha\end{bmatrix}.$$

**Нормализация вектора.** Единичный вектор $\hat{m}$ в направлении $\vec{v}$:

$$\hat{m} = \frac{1}{|\vec{v}|}\,\vec{v}, \qquad |\vec{v}| = \sqrt{v_1^2 + v_2^2 + v_3^2}.$$

**Пример.** Для $\vec{v} = (1, 2, -1)$: $|\vec{v}| = \sqrt{1+4+1} = \sqrt{6}$, поэтому $\hat{m} = \bigl(\tfrac{1}{\sqrt{6}}, \tfrac{2}{\sqrt{6}}, -\tfrac{1}{\sqrt{6}}\bigr)$.
