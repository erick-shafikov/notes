**Нулевое пространство** (null space) матрицы $A$ размера $m\times n$ — это множество всех решений однородной системы $A\vec{x}=\vec{0}$:

$$N(A) = \{\vec{x}\in\mathbb{R}^n \mid A\vec{x}=\vec{0}\}.$$

$N(A)$ является подпространством $\mathbb{R}^n$: нулевой вектор $\vec{0}\in N(A)$; если $v_1, v_2\in N(A)$, то $A(v_1+v_2)=Av_1+Av_2=\vec{0}$; если $v\in N(A)$, то $A(cv)=cAv=\vec{0}$.

**Пример.** Матрица $A$ размера $3\times4$:

$$A = \begin{bmatrix}1&1&1&1\\1&2&3&4\\4&3&2&1\end{bmatrix}.$$

После приведения к ступенчатому виду: $x_1 = x_3+2x_4$, $x_2=-2x_3-3x_4$. Переменные $x_3, x_4$ — свободные, решение:

$$\vec{x} = x_3\begin{bmatrix}1\\-2\\1\\0\end{bmatrix} + x_4\begin{bmatrix}2\\-3\\0\\1\end{bmatrix}, \quad N(A) = \operatorname{span}\!\left(\begin{bmatrix}1\\-2\\1\\0\end{bmatrix},\ \begin{bmatrix}2\\-3\\0\\1\end{bmatrix}\right).$$

**Пространство столбцов** (column space) матрицы $A=[v_1,\ldots,v_n]$ — линейная оболочка её столбцов:

$$C(A) = \operatorname{span}(v_1,\ldots,v_n) = \{A\vec{x}\mid \vec{x}\in\mathbb{R}^n\}.$$

$C(A)$ — подпространство $\mathbb{R}^m$. Система $A\vec{x}=\vec{b}$ совместна тогда и только тогда, когда $\vec{b}\in C(A)$. Если $\vec{b}\notin C(A)$ — решений нет; если $\vec{b}\in C(A)$ — решение существует.

**Связь $N(A)$ с линейной зависимостью.** Если $N(A)=\{\vec{0}\}$, то столбцы $v_1,\ldots,v_n$ линейно независимы: равенство $x_1 v_1+\cdots+x_n v_n=\vec{0}$ имеет только тривиальное решение.
