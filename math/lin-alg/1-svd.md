# SVD и ортогональная проекция

## Задача ортогональной проекции

Основная цель — сокращение признакового пространства: исходные объекты описаны $n$ признаками, задача — перейти к $m \ll n$ новым признакам, потеряв как можно меньше информации.

Пусть дана матрица $F \in \mathbb{R}^{l \times n}$ и ортогональная матрица $U \in \mathbb{R}^{m \times n}$, задающая новый базис из $m \leq n$ векторов. Проекция строк $F$ на новый базис:

$$G = F U^T, \qquad G \in \mathbb{R}^{l \times m}$$

Обратное восстановление (приближение исходных строк через проекции):

$$\hat{F} = G U = F U^T U$$

Задача оптимального выбора $U$: минимизировать суммарную ошибку восстановления:

$$\|F - F U^T U\|^2 \to \min_U$$

Оптимальные строки $U$ — это $m$ собственных векторов матрицы $F^T F$ с наибольшими собственными значениями $\lambda_1 \geq \cdots \geq \lambda_m$. Интуиция: $\lambda_j$ — это дисперсия данных вдоль направления $u_j$; чем больше $\lambda_j$, тем больше информации несёт проекция на $u_j$. Отбрасывая направления с малыми $\lambda_j$, мы теряем минимум — поэтому выгодно оставить именно те $m$ направлений, где разброс наибольший.

## Сингулярное разложение (SVD)

Любую матрицу $F \in \mathbb{R}^{l \times n}$ можно представить в виде:

$$F = V D U^T$$

где:
- $U \in \mathbb{R}^{n \times n}$ — матрица **правых сингулярных векторов**. Её столбцы $u_1, \ldots, u_n$ — ортонормированный базис в пространстве строк $F$ (входное пространство размерности $n$): $U^T U = I_n$. Правые векторы отвечают на вопрос «в каких направлениях исходного пространства данные имеют наибольший разброс».

- $V \in \mathbb{R}^{l \times l}$ — матрица **левых сингулярных векторов**. Её столбцы $v_1, \ldots, v_l$ — ортонормированный базис в пространстве столбцов $F$ (выходное пространство размерности $l$): $V^T V = I_l$. Левые векторы — это направления, в которые «разворачиваются» образы $u_j$ после умножения на $F$: $F u_j = \sigma_j v_j$.

- $D \in \mathbb{R}^{l \times n}$ — прямоугольная диагональная матрица с **сингулярными значениями** $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_m \geq 0$ на диагонали, где $m = \min(l, n)$. Каждое $\sigma_j$ показывает, насколько сильно $F$ «растягивает» направление $u_j$: чем больше $\sigma_j$, тем значимее это направление.

Геометрический смысл произведения $VDU^T$: сначала $U^T$ поворачивает вектор в базис правых сингулярных векторов, затем $D$ масштабирует каждую координату на $\sigma_j$, затем $V$ поворачивает результат в выходное пространство.

## Связь SVD с матрицей Грама

Матрица Грама $F^T F$ (размер $n \times n$) кодирует попарные скалярные произведения признаков: элемент $(i,j)$ равен $\langle f_i, f_j \rangle$, где $f_i$ — $i$-й столбец $F$. Она нужна для двух вещей: во-первых, её собственные векторы дают оптимальные направления проекции (правые сингулярные векторы $F$), во-вторых, её собственные значения $\lambda_j = \sigma_j^2$ показывают, сколько дисперсии данных приходится на каждое из этих направлений.

Произведение $F^T F$ раскладывается через SVD:

$$F^T F = (V D U^T)^T (V D U^T) = U D^T V^T V D U^T = U D^T D U^T$$

Произведение $D^T D$ — квадратная диагональная матрица размера $n \times n$ с $\lambda_j = \sigma_j^2$ на диагонали:

$$F^T F = U \begin{bmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_m \\ & & & 0 \end{bmatrix} U^T$$

Это в точности **спектральное разложение** матрицы $F^T F$: столбцы $U$ — её собственные векторы, $\lambda_j$ — собственные значения:

$$F^T F \cdot u_j = \lambda_j u_j$$

**Почему $\lambda_j$ — это дисперсия вдоль $u_j$.** Дисперсия проекций строк $F$ на единичный вектор $u$ равна:

$$\text{Var}(u) = \frac{1}{l}\sum_{k=1}^l (f_k^T u)^2 = \frac{1}{l}\|Fu\|^2 = \frac{1}{l} u^T F^T F\, u$$

Подставляя собственный вектор $u_j$ (для которого $F^TF\, u_j = \lambda_j u_j$):

$$\text{Var}(u_j) = \frac{1}{l}\, u_j^T \lambda_j u_j = \frac{\lambda_j}{l}$$

Дисперсия вдоль $u_j$ пропорциональна $\lambda_j$. Через SVD это видно напрямую: $Fu_j = VDU^Tu_j = \sigma_j v_j$, поэтому $\|Fu_j\|^2 = \sigma_j^2 = \lambda_j$ — сингулярное значение $\sigma_j$ есть «длина» образа базисного вектора $u_j$, а её квадрат и есть суммарный разброс проекций вдоль этого направления.

**Вывод:** правые сингулярные векторы матрицы $F$ совпадают с собственными векторами $F^T F$, а собственные значения равны квадратам сингулярных значений: $\lambda_j = \sigma_j^2$. Если вместо $F^TF$ используется нормированная версия $\frac{1}{l}F^TF$ (как в PCA), собственные векторы те же, а собственные значения масштабируются: $\lambda_j^{\text{норм}} = \sigma_j^2 / l$.

Аналогично, $F F^T = V D D^T V^T$ — левые сингулярные векторы суть собственные векторы $F F^T$ с теми же ненулевыми собственными значениями.

## Псевдообратная матрица

Для матрицы полного ранга псевдообратная определяется как $F^+ = (F^T F)^{-1} F^T$. Через SVD:

$$F^+ = (U D^T D U^T)^{-1} U D^T V^T = U (D^T D)^{-1} D^T V^T$$

Диагональные элементы $(D^T D)^{-1} D^T$ равны $1/\sigma_j$ для ненулевых сингулярных значений, и для вектора $y$ решение системы $F^T F\, x = F^T y$ выражается как:

$$x^* = F^+ y = \sum_{j=1}^m \frac{1}{\sigma_j}\, u_j\, (v_j^T y)$$

Каждое слагаемое — проекция $y$ на $v_j$, масштабированная на $1/\sigma_j$ и направленная вдоль $u_j$. При малых $\sigma_j$ соответствующее слагаемое огромно — решение нестабильно.

## Численный пример

Возьмём матрицу $F = \begin{bmatrix}2 & 1 \\ 1 & 2\end{bmatrix}$. Она симметрична — это упрощает вычисления и позволяет сосредоточиться на структуре разложения, не теряясь в арифметике.

**Шаг 1. Вычислить $F^T F$.**

$$F^T F = \begin{bmatrix}2&1\\1&2\end{bmatrix}^T\begin{bmatrix}2&1\\1&2\end{bmatrix} = \begin{bmatrix}2&1\\1&2\end{bmatrix}\begin{bmatrix}2&1\\1&2\end{bmatrix} = \begin{bmatrix}4+1&2+2\\2+2&1+4\end{bmatrix} = \begin{bmatrix}5&4\\4&5\end{bmatrix}$$

**Шаг 2. Найти собственные значения $F^T F$ — они дадут $\sigma_j^2$.**

$$\det(F^TF - \lambda I) = \det\begin{bmatrix}5-\lambda & 4 \\ 4 & 5-\lambda\end{bmatrix} = (5-\lambda)^2 - 16 = 0$$

$$5 - \lambda = \pm 4 \quad\Rightarrow\quad \lambda_1 = 9,\quad \lambda_2 = 1$$

Сингулярные значения: $\sigma_1 = \sqrt{9} = 3$, $\sigma_2 = \sqrt{1} = 1$.

**Шаг 3. Найти правые сингулярные векторы — столбцы $U$.**

Правые сингулярные векторы — это собственные векторы $F^TF$.

Для $\lambda_1 = 9$:

$$(F^TF - 9I)\,u = \begin{bmatrix}-4&4\\4&-4\end{bmatrix}u = 0 \quad\Rightarrow\quad -4u_1 + 4u_2 = 0 \quad\Rightarrow\quad u_1 = u_2$$

Нормируем: $\|u\|^2 = u_1^2 + u_2^2 = 2u_1^2 = 1 \;\Rightarrow\; u_1 = \frac{1}{\sqrt{2}}$, итого:

$$\tilde{u}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}$$

Для $\lambda_2 = 1$:

$$(F^TF - I)\,u = \begin{bmatrix}4&4\\4&4\end{bmatrix}u = 0 \quad\Rightarrow\quad 4u_1 + 4u_2 = 0 \quad\Rightarrow\quad u_1 = -u_2$$

$$\tilde{u}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}$$

Проверка ортогональности: $\tilde{u}_1^T \tilde{u}_2 = \frac{1}{2}(1 \cdot 1 + 1 \cdot (-1)) = 0$ ✓

Матрица $U$ составляется, просто поставив найденные векторы столбцами рядом:

$$U = \bigl[\,\tilde{u}_1 \;\big|\; \tilde{u}_2\,\bigr] = \left[\;\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix} \;\Bigg|\; \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}\;\right] = \frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$$

Первый столбец — $\tilde{u}_1$, второй — $\tilde{u}_2$. Множитель $\frac{1}{\sqrt{2}}$ вынесен за скобку, так как он общий для обоих векторов.

**Шаг 4. Найти левые сингулярные векторы — столбцы $V$.**

Левые сингулярные векторы вычисляются по формуле $v_j = \dfrac{F\,\tilde{u}_j}{\sigma_j}$:

$$v_1 = \frac{1}{\sigma_1} F\,\tilde{u}_1 = \frac{1}{3}\begin{bmatrix}2&1\\1&2\end{bmatrix}\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix} = \frac{1}{3\sqrt{2}}\begin{bmatrix}2\cdot1+1\cdot1\\1\cdot1+2\cdot1\end{bmatrix} = \frac{1}{3\sqrt{2}}\begin{bmatrix}3\\3\end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}$$

$$v_2 = \frac{1}{\sigma_2} F\,\tilde{u}_2 = \frac{1}{1}\begin{bmatrix}2&1\\1&2\end{bmatrix}\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix}2-1\\1-2\end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}$$

$$V = \frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$$

Здесь $V = U$ — следствие симметричности $F$: для симметричной матрицы левые и правые сингулярные векторы совпадают. Для несимметричной матрицы $V \neq U$ в общем случае.

**Шаг 5. Собрать разложение $F = VDU^T$ и проверить.**

Матрица $D$ составляется из сингулярных значений, найденных на шаге 2 ($\sigma_1 = 3$, $\sigma_2 = 1$), которые ставятся на диагональ в порядке убывания:

$$D = \begin{bmatrix}\sigma_1&0\\0&\sigma_2\end{bmatrix} = \begin{bmatrix}3&0\\0&1\end{bmatrix}$$

Вычисляем $VD$:

$$VD = \frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix}\begin{bmatrix}3&0\\0&1\end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix}3&1\\3&-1\end{bmatrix}$$

Умножаем на $U^T$:

$$(VD)U^T = \frac{1}{\sqrt{2}}\begin{bmatrix}3&1\\3&-1\end{bmatrix} \cdot \frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix} = \frac{1}{2}\begin{bmatrix}3\cdot1+1\cdot1 & 3\cdot1+1\cdot(-1)\\3\cdot1+(-1)\cdot1 & 3\cdot1+(-1)\cdot(-1)\end{bmatrix} = \frac{1}{2}\begin{bmatrix}4&2\\2&4\end{bmatrix} = \begin{bmatrix}2&1\\1&2\end{bmatrix} \checkmark$$

**Шаг 6. Проверить связь $Fu_j = \sigma_j v_j$ для каждого сингулярного вектора.**

$$F\,\tilde{u}_1 = \begin{bmatrix}2&1\\1&2\end{bmatrix}\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix}3\\3\end{bmatrix} = 3 \cdot \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix} = \sigma_1 v_1 \checkmark$$

$$F\,\tilde{u}_2 = \begin{bmatrix}2&1\\1&2\end{bmatrix}\frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix} = 1 \cdot \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix} = \sigma_2 v_2 \checkmark$$

Это и есть ключевое свойство SVD: $F$ переводит каждый правый сингулярный вектор в соответствующий левый, масштабируя его на $\sigma_j$. Направление $\tilde{u}_1 = [1,1]^T/\sqrt{2}$ (диагональ) растягивается втрое, направление $\tilde{u}_2 = [1,-1]^T/\sqrt{2}$ (антидиагональ) остаётся без изменений.

## Численный пример: несимметричная матрица

Возьмём матрицу $F = \begin{bmatrix}0 & 3 \\ 2 & 0\end{bmatrix}$. Она несимметрична ($F_{12} = 3 \neq F_{21} = 2$), поэтому ожидаем $V \neq U$.

**Шаг 1. Вычислить $F^T F$.**

$$F^T F = \begin{bmatrix}0&2\\3&0\end{bmatrix}\begin{bmatrix}0&3\\2&0\end{bmatrix} = \begin{bmatrix}4&0\\0&9\end{bmatrix}$$

**Шаг 2. Найти сингулярные значения.**

Матрица $F^TF$ диагональна, собственные значения стоят на диагонали: $\lambda_1 = 9,\ \lambda_2 = 4$.

$$\sigma_1 = \sqrt{9} = 3, \qquad \sigma_2 = \sqrt{4} = 2$$

**Шаг 3. Найти правые сингулярные векторы — столбцы $U$.**

Для $\lambda_1 = 9$:

$$(F^TF - 9I)\,u = \begin{bmatrix}-5&0\\0&0\end{bmatrix}u = 0 \quad\Rightarrow\quad u_1 = 0 \quad\Rightarrow\quad \tilde{u}_1 = \begin{bmatrix}0\\1\end{bmatrix}$$

Для $\lambda_2 = 4$:

$$(F^TF - 4I)\,u = \begin{bmatrix}0&0\\0&5\end{bmatrix}u = 0 \quad\Rightarrow\quad u_2 = 0 \quad\Rightarrow\quad \tilde{u}_2 = \begin{bmatrix}1\\0\end{bmatrix}$$

$$U = \begin{bmatrix}0&1\\1&0\end{bmatrix}$$

**Шаг 4. Найти левые сингулярные векторы — столбцы $V$.**

$$v_1 = \frac{1}{\sigma_1}F\,\tilde{u}_1 = \frac{1}{3}\begin{bmatrix}0&3\\2&0\end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix} = \frac{1}{3}\begin{bmatrix}3\\0\end{bmatrix} = \begin{bmatrix}1\\0\end{bmatrix}$$

$$v_2 = \frac{1}{\sigma_2}F\,\tilde{u}_2 = \frac{1}{2}\begin{bmatrix}0&3\\2&0\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} = \frac{1}{2}\begin{bmatrix}0\\2\end{bmatrix} = \begin{bmatrix}0\\1\end{bmatrix}$$

$$V = \begin{bmatrix}1&0\\0&1\end{bmatrix} = I$$

**Шаг 5. Собрать разложение $F = VDU^T$ и проверить.**

$$D = \begin{bmatrix}3&0\\0&2\end{bmatrix}$$

$$VDU^T = I\begin{bmatrix}3&0\\0&2\end{bmatrix}\begin{bmatrix}0&1\\1&0\end{bmatrix} = \begin{bmatrix}3&0\\0&2\end{bmatrix}\begin{bmatrix}0&1\\1&0\end{bmatrix} = \begin{bmatrix}0&3\\2&0\end{bmatrix} = F \checkmark$$

**Шаг 6. Проверить $F u_j = \sigma_j v_j$.**

$$F\,\tilde{u}_1 = \begin{bmatrix}0&3\\2&0\end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}3\\0\end{bmatrix} = 3\begin{bmatrix}1\\0\end{bmatrix} = \sigma_1 v_1 \checkmark$$

$$F\,\tilde{u}_2 = \begin{bmatrix}0&3\\2&0\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}0\\2\end{bmatrix} = 2\begin{bmatrix}0\\1\end{bmatrix} = \sigma_2 v_2 \checkmark$$

Здесь $V = I \neq U$ — несимметричность $F$ приводит к тому, что входное и выходное пространства ориентированы по-разному: $F$ отображает второй стандартный вектор в первый (масштабируя на 3) и первый — во второй (масштабируя на 2). Для симметричной матрицы такого «перекрёста» нет, поэтому $V = U$.
