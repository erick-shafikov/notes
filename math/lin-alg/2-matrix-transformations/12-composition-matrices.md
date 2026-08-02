**Композиция преобразований.** Для отображений $S: X \to Y$ и $T: Y \to Z$ их композиция $T \circ S: X \to Z$ определяется как $(T \circ S)(\vec{x}) = T(S(\vec{x}))$.

**Теорема.** Если $S$ и $T$ линейны, то $T \circ S$ линейно.

*Доказательство:* $(T \circ S)(\vec{x}+\vec{y}) = T(S(\vec{x}+\vec{y})) = T(S(\vec{x})+S(\vec{y})) = T(S(\vec{x}))+T(S(\vec{y})) = (T\circ S)(\vec{x})+(T\circ S)(\vec{y})$. Аналогично $(T\circ S)(c\vec{x}) = c(T\circ S)(\vec{x})$.

**Матрица композиции.** Пусть $S(\vec{x}) = A\vec{x}$ (матрица $m \times n$) и $T(\vec{x}) = B\vec{x}$ (матрица $\ell \times m$). Тогда:

$$(T \circ S)(\vec{x}) = B(A\vec{x}) = (BA)\vec{x} = C\vec{x}, \quad C = BA.$$

Порядок множителей важен: $C = BA$, а не $AB$.

**Умножение матриц по столбцам.** Если $A$ — матрица размера $m \times n$ и $B = [b_1\ b_2\ \cdots\ b_k]$ — матрица $n \times k$, то:

$$AB = [Ab_1\ Ab_2\ \cdots\ Ab_k].$$

**Пример.** $A = \begin{bmatrix}1&-1&2\\0&-2&1\end{bmatrix}$, $B$ — матрица $3 \times 4$. Произведение $AB$ вычисляется применением $A$ к каждому столбцу $B$:

$$AB = \begin{bmatrix}5&2&0&6\\-1&1&-2&4\end{bmatrix}.$$

**Согласование размеров.** Если $S: \mathbb{R}^4 \to \mathbb{R}^3$ (матрица $3\times4$) и $T: \mathbb{R}^3 \to \mathbb{R}^2$ (матрица $2\times3$), то $T \circ S: \mathbb{R}^4 \to \mathbb{R}^2$ имеет матрицу $2\times4$. Обратная же композиция $S \circ T: \mathbb{R}^2 \to \mathbb{R}^2$ потребовала бы $S: \mathbb{R}^2 \to \mathbb{R}^3$ — это другое отображение.
