**Ортонормированный базис.** Базис $B = \{\vec{v}_1, \vec{v}_2, \ldots, \vec{v}_k\}$ пространства $V$ называется ортонормированным, если все векторы имеют единичную длину и попарно ортогональны:

$$\vec{v}_i \cdot \vec{v}_j = \begin{cases}1, & i = j,\\ 0, & i \neq j.\end{cases}$$

Стандартный базис $\{e_1, \ldots, e_n\}$ пространства $\mathbb{R}^n$ ортонормирован. Пример нестандартного ортонормированного базиса: $\vec{v}_1 = \tfrac{1}{3}(1,2,2)^T$, $\vec{v}_2 = \tfrac{1}{3}(2,1,-2)^T$ — проверяем: $|\vec{v}_1| = |\vec{v}_2| = 1$, $\vec{v}_1 \cdot \vec{v}_2 = \tfrac{1}{9}(2+2-4) = 0$.

**Координаты в ортонормированном базисе.** Если $B = \{\vec{v}_1, \ldots, \vec{v}_k\}$ — ортонормированный базис $V$ и $\vec{x} \in V$, то $\vec{x} = c_1\vec{v}_1 + \cdots + c_k\vec{v}_k$. Умножив скалярно на $\vec{v}_i$, получим $\vec{v}_i \cdot \vec{x} = c_i$, поскольку все остальные слагаемые обнуляются. Таким образом,

$$[\vec{x}]_B = \begin{pmatrix}\vec{v}_1 \cdot \vec{x}\\ \vdots\\ \vec{v}_k \cdot \vec{x}\end{pmatrix} = C^T \vec{x},$$

где $C = [\vec{v}_1\,|\cdots|\,\vec{v}_k]$. Для ортонормированного базиса $C^T C = I$, поэтому $C^{-1} = C^T$ — обратная матрица находится просто транспонированием, без формулы обращения.

**Проекция с ортонормированным базисом.** Если столбцы $A$ образуют ортонормированный базис подпространства $V$, то $A^T A = I$, и формула проекции упрощается до

$$\operatorname{Proj}_V \vec{x} = A A^T \vec{x} = (\vec{v}_1 \cdot \vec{x})\vec{v}_1 + (\vec{v}_2 \cdot \vec{x})\vec{v}_2 + \cdots + (\vec{v}_k \cdot \vec{x})\vec{v}_k.$$

В общем случае (неортонормированный базис) формула $A(A^T A)^{-1}A^T\vec{x}$ требует обращения $A^T A$, тогда как при ортонормированном базисе достаточно $AA^T\vec{x}$.

**Метод Грама–Шмидта.** По произвольному базису $\{\vec{v}_1, \ldots, \vec{v}_k\}$ подпространства $V$ строится ортонормированный базис $\{\vec{u}_1, \ldots, \vec{u}_k\}$ того же пространства.

Шаг 1. $\vec{u}_1 = \dfrac{\vec{v}_1}{|\vec{v}_1|}$.

Шаг 2. Вычитаем из $\vec{v}_2$ его проекцию на $\operatorname{span}(\vec{u}_1)$:
$$\vec{y}_2 = \vec{v}_2 - (\vec{v}_2 \cdot \vec{u}_1)\vec{u}_1, \qquad \vec{u}_2 = \frac{\vec{y}_2}{|\vec{y}_2|}.$$

Шаг 3. Вычитаем из $\vec{v}_3$ проекции на $\operatorname{span}(\vec{u}_1, \vec{u}_2)$:
$$\vec{y}_3 = \vec{v}_3 - (\vec{v}_3 \cdot \vec{u}_1)\vec{u}_1 - (\vec{v}_3 \cdot \vec{u}_2)\vec{u}_2, \qquad \vec{u}_3 = \frac{\vec{y}_3}{|\vec{y}_3|}.$$

Шаг $i$ (общий). $\vec{y}_i = \vec{v}_i - \sum_{j=1}^{i-1}(\vec{v}_i \cdot \vec{u}_j)\vec{u}_j$, $\quad \vec{u}_i = \vec{y}_i / |\vec{y}_i|$.

На каждом шаге $\vec{y}_i$ ортогонален всем предыдущим $\vec{u}_j$, а деление нормирует вектор до единичной длины.

**Пример в $\mathbb{R}^3$.** $V = \{(x_1,x_2,x_3) \mid x_1+x_2+x_3 = 0\}$, исходный базис $\vec{v}_1 = (-1,1,0)^T$, $\vec{v}_2 = (-1,0,1)^T$.

Шаг 1: $|\vec{v}_1| = \sqrt{2}$, $\vec{u}_1 = \tfrac{1}{\sqrt{2}}(-1,1,0)^T$.

Шаг 2: $\vec{v}_2 \cdot \vec{u}_1 = \tfrac{1}{\sqrt{2}}(1+0+0) = \tfrac{1}{\sqrt{2}}$. Тогда
$$\vec{y}_2 = (-1,0,1)^T - \tfrac{1}{\sqrt{2}} \cdot \tfrac{1}{\sqrt{2}}(-1,1,0)^T = (-1,0,1)^T - \tfrac{1}{2}(-1,1,0)^T = \begin{pmatrix}-1/2\\-1/2\\1\end{pmatrix}.$$
$|\vec{y}_2| = \sqrt{1/4+1/4+1} = \sqrt{3/2}$, $\vec{u}_2 = \tfrac{1}{\sqrt{3/2}}\begin{pmatrix}-1/2\\-1/2\\1\end{pmatrix} = \tfrac{1}{\sqrt{6}}\begin{pmatrix}-1\\-1\\2\end{pmatrix}$.

Ортонормированный базис $V$: $\left\{\,\tfrac{1}{\sqrt{2}}\begin{pmatrix}-1\\1\\0\end{pmatrix},\ \tfrac{1}{\sqrt{6}}\begin{pmatrix}-1\\-1\\2\end{pmatrix}\right\}$.

**Пример в $\mathbb{R}^4$.** $V = \operatorname{span}(\vec{v}_1, \vec{v}_2, \vec{v}_3)$, где $\vec{v}_1 = (0,0,1,1)^T$, $\vec{v}_2 = (0,1,1,0)^T$, $\vec{v}_3 = (1,1,0,0)^T$.

Шаг 1: $|\vec{v}_1| = \sqrt{2}$, $\vec{u}_1 = \tfrac{1}{\sqrt{2}}(0,0,1,1)^T$.

Шаг 2: $\vec{v}_2 \cdot \vec{u}_1 = \tfrac{1}{\sqrt{2}}$, $\vec{y}_2 = (0,1,1,0)^T - \tfrac{1}{2}(0,0,1,1)^T = (0,1,\tfrac{1}{2},-\tfrac{1}{2})^T$, $|\vec{y}_2| = \sqrt{3/2}$, $\vec{u}_2 = \sqrt{\tfrac{2}{3}}(0,1,\tfrac{1}{2},-\tfrac{1}{2})^T$.

Шаг 3: $\vec{y}_3 = \vec{v}_3 - (\vec{v}_3 \cdot \vec{u}_1)\vec{u}_1 - (\vec{v}_3 \cdot \vec{u}_2)\vec{u}_2$. Вычисляя, получаем $\vec{u}_3 = \tfrac{1}{2\sqrt{3}}(3,-1,-1,1)^T$ (после нормировки $|\vec{y}_3| = 2\sqrt{3}$).
