**Обратная матрица.** Для квадратной матрицы $A$ размера $n \times n$ обратная матрица $A^{-1}$ удовлетворяет $AA^{-1} = A^{-1}A = I_n$.

**Метод нахождения $A^{-1}$.** Записываем расширенную матрицу $[A \mid I]$ и приводим её к виду $[I \mid A^{-1}]$ элементарными преобразованиями строк. Каждое такое преобразование соответствует умножению слева на элементарную матрицу $S_i$. После серии шагов $S_k \cdots S_2 S_1 A = I$, откуда $A^{-1} = S_k \cdots S_2 S_1$.

**Пример.** $A = \begin{bmatrix}-1&1&1\\-1&2&3\\1&1&4\end{bmatrix}$.

$$[A \mid I] \to \begin{bmatrix}1&-1&-1&\mid&1&0&0\\0&1&2&\mid&1&1&0\\0&2&5&\mid&-1&0&1\end{bmatrix} \to \begin{bmatrix}1&0&1&\mid&2&1&0\\0&1&2&\mid&1&1&0\\0&0&1&\mid&-3&-2&1\end{bmatrix} \to \begin{bmatrix}1&0&0&\mid&5&3&-1\\0&1&0&\mid&7&5&-2\\0&0&1&\mid&-3&-2&1\end{bmatrix}.$$

Поэтому $A^{-1} = \begin{bmatrix}5&3&-1\\7&5&-2\\-3&-2&1\end{bmatrix}$.

**Формула для матрицы $2 \times 2$.** Для $A = \begin{bmatrix}a&b\\c&d\end{bmatrix}$:

$$\det(A) = ad - bc, \qquad A^{-1} = \frac{1}{\det(A)}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}.$$

Если $\det(A) = 0$, матрица необратима (вырожденная).

**Пример.** $B = \begin{bmatrix}1&2\\3&4\end{bmatrix}$, $|B| = 1\cdot4 - 2\cdot3 = -2$:

$$B^{-1} = -\frac{1}{2}\begin{bmatrix}4&-2\\-3&1\end{bmatrix} = \begin{bmatrix}-2&1\\3/2&-1/2\end{bmatrix}.$$

**Определитель** матрицы $3 \times 3$ вычисляется разложением по первой строке:

$$\det(A) = a_{11}\begin{vmatrix}a_{22}&a_{23}\\a_{32}&a_{33}\end{vmatrix} - a_{12}\begin{vmatrix}a_{21}&a_{23}\\a_{31}&a_{33}\end{vmatrix} + a_{13}\begin{vmatrix}a_{21}&a_{22}\\a_{31}&a_{32}\end{vmatrix}.$$

**Правило Сарруса** (для $3 \times 3$):

$$\begin{vmatrix}a&b&c\\d&e&f\\g&h&i\end{vmatrix} = aei + bfg + cdh - ceg - bdi - afh.$$
