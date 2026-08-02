**Собственные векторы и собственные значения.** Ненулевой вектор $\vec{v} \in \mathbb{R}^n$ называется собственным вектором матрицы $A$ (размера $n \times n$), если существует число $\lambda$ такое, что

$$A\vec{v} = \lambda\vec{v}.$$

Число $\lambda$ называется собственным значением, соответствующим $\vec{v}$. Геометрически: линейное преобразование $T(\vec{x}) = A\vec{x}$ лишь растягивает (или сжимает, или отражает) $\vec{v}$, не меняя его направления.

**Характеристический многочлен.** Из $A\vec{v} = \lambda\vec{v}$ при $\vec{v} \neq \vec{0}$ следует $(\lambda I_n - A)\vec{v} = \vec{0}$, то есть матрица $B = \lambda I_n - A$ имеет ненулевое нулевое пространство. Это возможно тогда и только тогда, когда

$$\det(\lambda I_n - A) = 0.$$

Левая часть — многочлен от $\lambda$ степени $n$, называемый характеристическим многочленом матрицы $A$. Его корни — собственные значения.

**Пример ($2 \times 2$).** $A = \begin{bmatrix}1&2\\4&3\end{bmatrix}$.

$$\det(\lambda I_2 - A) = \det\!\begin{bmatrix}\lambda-1&-2\\-4&\lambda-3\end{bmatrix} = (\lambda-1)(\lambda-3)-8 = \lambda^2 - 4\lambda - 5 = 0.$$

Собственные значения: $\lambda = 5$ и $\lambda = -1$.

**Собственное подпространство** $E_\lambda = N(\lambda I_n - A)$ — нулевое пространство матрицы $\lambda I_n - A$, то есть множество всех собственных векторов, соответствующих $\lambda$, вместе с нулевым вектором.

Для $\lambda = 5$: $5I - A = \begin{bmatrix}4&-2\\-4&2\end{bmatrix} \xrightarrow{\text{rref}} \begin{bmatrix}1&-1/2\\0&0\end{bmatrix}$, откуда $v_1 = \tfrac{1}{2}v_2$. Таким образом, $E_5 = \operatorname{span}\!\begin{pmatrix}1\\2\end{pmatrix}$.

Для $\lambda = -1$: $-I - A = \begin{bmatrix}-2&-2\\-4&-4\end{bmatrix} \xrightarrow{\text{rref}} \begin{bmatrix}1&1\\0&0\end{bmatrix}$, откуда $v_1 = -v_2$. Таким образом, $E_{-1} = \operatorname{span}\!\begin{pmatrix}1\\-1\end{pmatrix}$.

**Пример ($3 \times 3$).** $A = \begin{bmatrix}-1&2&2\\2&2&-1\\2&-1&2\end{bmatrix}$.

$$\det(\lambda I - A) = \lambda^3 - 3\lambda^2 - 9\lambda + 27 = (\lambda-3)^2(\lambda+3) = 0.$$

Собственные значения: $\lambda = 3$ (кратное) и $\lambda = -3$.

При $\lambda = 3$: $3I - A = \begin{bmatrix}4&-2&-2\\-2&1&1\\-2&1&1\end{bmatrix} \xrightarrow{\text{rref}} \begin{bmatrix}1&-1/2&-1/2\\0&0&0\\0&0&0\end{bmatrix}$, откуда $v_1 = \tfrac{1}{2}v_2 + \tfrac{1}{2}v_3$. Собственное подпространство двумерно: $E_3 = \operatorname{span}\!\left(\begin{pmatrix}1\\2\\0\end{pmatrix},\begin{pmatrix}1\\0\\2\end{pmatrix}\right)$.

При $\lambda = -3$: $-3I - A = \begin{bmatrix}-2&-2&-2\\-2&5&1\\-2&1&-5\end{bmatrix} \xrightarrow{\text{rref}} \begin{bmatrix}1&0&2\\0&1&-1\\0&0&0\end{bmatrix}$, откуда $v_1 = -2v_3$, $v_2 = v_3$. Тогда $E_{-3} = \operatorname{span}\!\begin{pmatrix}-2\\1\\1\end{pmatrix}$.

**Собственный базис и диагонализация.** Если $A$ имеет $n$ линейно независимых собственных векторов $\vec{v}_1, \ldots, \vec{v}_n$ с собственными значениями $\lambda_1, \ldots, \lambda_n$, то в базисе $B = \{\vec{v}_1, \ldots, \vec{v}_n\}$ матрица преобразования принимает диагональный вид:

$$D = C^{-1}AC = \begin{bmatrix}\lambda_1 & & \\ & \ddots & \\ & & \lambda_n\end{bmatrix},$$

где $C = [\vec{v}_1\,|\cdots|\,\vec{v}_n]$. Это следует из того, что $[T(\vec{v}_i)]_B = \lambda_i \vec{e}_i$ — $i$-й столбец $D$ равен $\lambda_i \vec{e}_i$. Эквивалентно, $A = CDC^{-1}$. Наличие такого базиса означает, что матрица $A$ диагонализируема, и работа с ней в собственном базисе становится особенно простой.
