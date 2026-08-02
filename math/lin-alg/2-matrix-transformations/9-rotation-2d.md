**Поворот в $\mathbb{R}^2$** на угол $\alpha$ против часовой стрелки — линейное преобразование $\operatorname{Rot}_\alpha: \mathbb{R}^2 \to \mathbb{R}^2$.

Линейность: $\operatorname{Rot}_\alpha(\vec{x}+\vec{y}) = \operatorname{Rot}_\alpha(\vec{x}) + \operatorname{Rot}_\alpha(\vec{y})$ (поворот сохраняет векторное сложение) и $\operatorname{Rot}_\alpha(c\vec{x}) = c\operatorname{Rot}_\alpha(\vec{x})$.

Стандартная матрица строится по образам базисных векторов. Единичный вектор $e_1 = (1,0)$ при повороте на $\alpha$ переходит в $(\cos\alpha, \sin\alpha)$; вектор $e_2 = (0,1)$ — в $(-\sin\alpha, \cos\alpha)$:

$$A = \begin{bmatrix}\cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha\end{bmatrix}, \qquad \operatorname{Rot}_\alpha(\vec{x}) = A\vec{x}.$$

**Пример.** Поворот на $\alpha = 45°$:

$$A = \begin{bmatrix}\frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \\[4pt] \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}\end{bmatrix}.$$

Матрица поворота ортогональна: $A^T A = I$, то есть $A^{-1} = A^T$. Это означает, что обратное преобразование — поворот на $-\alpha$.
