**Матрица линейного преобразования в альтернативном базисе.** Пусть $T: \mathbb{R}^n \to \mathbb{R}^n$ — линейное преобразование, заданное стандартной матрицей $A$ ($T(\vec{x}) = A\vec{x}$), и пусть $B = \{\vec{v}_1, \ldots, \vec{v}_n\}$ — базис $\mathbb{R}^n$ с матрицей смены базиса $C = \begin{bmatrix}\vec{v}_1 & \cdots & \vec{v}_n\end{bmatrix}$. Тогда матрица $D$ того же преобразования в базисе $B$ (то есть удовлетворяющая $[T(\vec{x})]_B = D[\vec{x}]_B$) равна

$$\boxed{D = C^{-1}AC.}$$

Это следует из цепочки: $[\vec{x}]_B = C^{-1}\vec{x}$, $\vec{x} = C[\vec{x}]_B$, поэтому $[T(\vec{x})]_B = C^{-1}A\vec{x} = C^{-1}A C[\vec{x}]_B = D[\vec{x}]_B$.

Коммутативная диаграмма: $\vec{x} \xrightarrow{A} T(\vec{x})$ в стандартном базисе; $[\vec{x}]_B \xrightarrow{D} [T(\vec{x})]_B$ в базисе $B$; переходы между строками — умножение на $C$ (вниз) и $C^{-1}$ (вверх).

Обратное соотношение: $A = CDC^{-1}$.

**Пример.** $T: \mathbb{R}^2 \to \mathbb{R}^2$, $A = \begin{bmatrix}3&-2\\2&-2\end{bmatrix}$, $B = \left\{\begin{pmatrix}1\\2\end{pmatrix},\begin{pmatrix}2\\1\end{pmatrix}\right\}$.

Матрица смены базиса $C = \begin{bmatrix}1&2\\2&1\end{bmatrix}$, $\det(C) = -3$, $C^{-1} = -\tfrac{1}{3}\begin{bmatrix}1&-2\\-2&1\end{bmatrix}$.

$$D = C^{-1}AC = -\tfrac{1}{3}\begin{bmatrix}1&-2\\-2&1\end{bmatrix}\begin{bmatrix}3&-2\\2&-2\end{bmatrix}\begin{bmatrix}1&2\\2&1\end{bmatrix} = \begin{bmatrix}-1&0\\0&2\end{bmatrix}.$$

Матрица $D$ диагональна — это означает, что $\vec{v}_1$ и $\vec{v}_2$ являются собственными векторами $A$: $A\vec{v}_1 = -\vec{v}_1$, $A\vec{v}_2 = 2\vec{v}_2$.

Проверка на $\vec{x} = (1,-1)^T$: $T(\vec{x}) = A(1,-1)^T = (5,4)^T$. Координаты в $B$: $[\vec{x}]_B = C^{-1}(1,-1)^T = -\tfrac{1}{3}(3,-3)^T = (-1,1)^T$. Применяем $D$: $D(-1,1)^T = (1,2)^T$. Обратно: $C(1,2)^T = (5,4)^T$ ✓.

**Случай ортонормированного базиса.** Если столбцы $C$ образуют ортонормированный базис, то $C^T C = I$, то есть $C^{-1} = C^T$, и формула упрощается:

$$D = C^T A C.$$
