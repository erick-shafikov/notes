## Расширенная числовая прямая

**Расширенная числовая прямая** $\overline{\mathbb{R}}$ — множество $\mathbb{R} \cup \{-\infty, +\infty\}$, где $-\infty$ и $+\infty$ — два новых элемента, удовлетворяющих:

$$-\infty < x < +\infty \quad \text{для всех } x \in \mathbb{R}$$

Арифметика с бесконечностями (при $a \in \mathbb{R}$):

$$a + \infty = +\infty, \quad a - \infty = -\infty$$
$$a \cdot (+\infty) = \begin{cases} +\infty, & a > 0 \\ -\infty, & a < 0 \end{cases}$$
$$\frac{a}{+\infty} = 0$$

**Не определены:** $\infty - \infty$, $0 \cdot \infty$, $\frac{\infty}{\infty}$ — неопределённые формы.

Окрестности бесконечностей: окрестность $+\infty$ — любой промежуток вида $(M, +\infty)$, $M \in \mathbb{R}$; аналогично $(-\infty, M)$ для $-\infty$.

**Применение.** Запись $\lim_{n \to \infty} x_n = +\infty$ означает: $\forall M > 0\;\exists N:\;\forall n > N:\; x_n > M$. Это предел в $\overline{\mathbb{R}}$, но не в $\mathbb{R}$ (последовательность расходится в обычном смысле).

$\sup E = +\infty$ означает, что $E$ неограничено сверху; $\inf E = -\infty$ — неограничено снизу. Соглашение: $\sup \varnothing = -\infty$, $\inf \varnothing = +\infty$.
