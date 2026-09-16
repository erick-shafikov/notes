# ККТ: числовой пример — разделяющая прямая SVM

## Задача

Дана выборка из 4 точек (2 признака, 2 класса):

| $x_1$ | $x_2$ | $y$ |
|--------|--------|-----|
| 1      | 1      | +1  |
| 2      | 2      | +1  |
| 0      | 1      | −1  |
| 1      | 0      | −1  |

Найти разделяющую прямую $\langle\omega, x\rangle - b = 0$ с максимальной шириной полосы.

## Постановка

Линейно разделимый случай, задача:

$$\frac{1}{2}\|\omega\|^2 \to \min \qquad \text{при} \quad y_i(\langle\omega, x_i\rangle - b) \geq 1 \quad \forall\, i$$

Ограничения в форме $g_i \leq 0$:

$$g_i = 1 - y_i(\langle\omega, x_i\rangle - b) \leq 0, \quad i = 1,2,3,4$$

## Функция Лагранжа

$$\mathcal{L} = \frac{1}{2}\|\omega\|^2 + \sum_{i=1}^4 \lambda_i\bigl(1 - y_i(\langle\omega, x_i\rangle - b)\bigr)$$

## Условия стационарности

**По $\omega$:**

$$\omega = \sum_{i=1}^4 \lambda_i y_i x_i$$

**По $b$:**

$$\sum_{i=1}^4 \lambda_i y_i = 0$$

## Дополняющая нежёсткость

$$\lambda_i\bigl(1 - y_i(\langle\omega, x_i\rangle - b)\bigr) = 0 \quad \forall\, i$$

Объекты вдали от полосы имеют $\lambda_i = 0$. Предположим, что опорные векторы — это точки $(1,1)$ и $(1,0)$ (ближайшие к границе с каждой стороны), то есть $\lambda_1 \neq 0$, $\lambda_4 \neq 0$, $\lambda_2 = \lambda_3 = 0$.

## Нахождение $\lambda_i$

Из условия $\sum \lambda_i y_i = 0$:

$$\lambda_1 \cdot (+1) + \lambda_4 \cdot (-1) = 0 \quad \Longrightarrow \quad \lambda_1 = \lambda_4 =: \lambda$$

Из стационарности $\omega = \sum \lambda_i y_i x_i$:

$$\omega = \lambda \cdot (+1) \cdot \begin{pmatrix}1\\1\end{pmatrix} + \lambda \cdot (-1) \cdot \begin{pmatrix}1\\0\end{pmatrix} = \lambda\begin{pmatrix}0\\1\end{pmatrix}$$

Из условия активности ограничений на опорных векторах ($g_i = 0$):

$$y_1(\langle\omega, x_1\rangle - b) = 1 \quad \Longrightarrow \quad \langle\omega, (1,1)\rangle - b = 1 \quad \Longrightarrow \quad \lambda - b = 1$$

$$y_4(\langle\omega, x_4\rangle - b) = 1 \quad \Longrightarrow \quad -(\langle\omega, (1,0)\rangle - b) = 1 \quad \Longrightarrow \quad b = 1$$

Из первого уравнения: $\lambda = 1 + b = 2$.

## Результат

$$\lambda_1 = \lambda_4 = 2, \qquad \omega = 2\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}0\\2\end{pmatrix}, \qquad b = 1$$

Разделяющая прямая: $0 \cdot x_1 + 2 \cdot x_2 - 1 = 0$, то есть $x_2 = \frac{1}{2}$.

Ширина полосы: $L = \dfrac{2}{\|\omega\|} = \dfrac{2}{2} = 1$.

## Проверка

| Точка | $y_i(\langle\omega, x_i\rangle - b)$ | Тип |
|-------|--------------------------------------|-----|
| $(1,1),\ y=+1$ | $+(2\cdot1 - 1) = 1$ | опорный |
| $(2,2),\ y=+1$ | $+(2\cdot2 - 1) = 3 \geq 1$ | обычный, $\lambda_2=0$ ✓ |
| $(0,1),\ y=-1$ | $-(2\cdot1 - 1) = -1$, т.е. отступ $=1$ | опорный? нет, $\lambda_3=0$ — проверим |
| $(1,0),\ y=-1$ | $-(2\cdot0 - 1) = 1$ | опорный |

Точка $(0,1)$: отступ $= -(2\cdot1 - 1) = -1$ — знак неверен, значит $\lambda_3 = 0$ корректно (она не нарушает ограничение — её отступ равен 1 по модулю, но проверим: $y_3(\langle\omega,x_3\rangle - b) = (-1)(2\cdot1 - 1) = -1$... это нарушение). 

Пересмотр: опорные — $(1,1)$ и $(0,1)$ с $y=-1$. Тогда:

$$\omega = \lambda\cdot(+1)\cdot\begin{pmatrix}1\\1\end{pmatrix} + \lambda\cdot(-1)\cdot\begin{pmatrix}0\\1\end{pmatrix} = \lambda\begin{pmatrix}1\\0\end{pmatrix}$$

Из активности:

$$\langle\omega,(1,1)\rangle - b = 1 \Rightarrow \lambda - b = 1$$
$$-(\langle\omega,(0,1)\rangle - b) = 1 \Rightarrow b = 1, \quad \lambda = 2$$

$$\omega = \begin{pmatrix}2\\0\end{pmatrix}, \quad b = 1$$

Разделяющая прямая: $2x_1 - 1 = 0$, то есть $x_1 = \frac{1}{2}$. Ширина полосы $L = \frac{2}{2} = 1$.

**Итог:** SVM нашёл вертикальную разделяющую прямую $x_1 = \frac{1}{2}$, опорные векторы — $(1,1)$ и $(0,1)$.
