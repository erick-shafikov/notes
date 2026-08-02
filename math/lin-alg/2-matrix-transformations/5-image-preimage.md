**Прообраз** (preimage) множества $S \subseteq Y$ при отображении $T: X \to Y$ — это множество всех элементов $X$, отображающихся в $S$:

$$T^{-1}(S) = \{x \in X \mid T(x) \in S\}.$$

При этом $T(T^{-1}(S)) \subseteq S$: образ прообраза вложен в исходное множество.

**Образ** (image) линейного преобразования $T: \mathbb{R}^n \to \mathbb{R}^m$ — это множество всех возможных значений:

$$\operatorname{Im}(T) = \{T(\vec{x}) \mid \vec{x} \in \mathbb{R}^n\}.$$

Если $T(\vec{x}) = A\vec{x}$ и $A = [a_1\ a_2\ \cdots\ a_n]$, то

$$\operatorname{Im}(T) = \{A\vec{x} \mid \vec{x} \in \mathbb{R}^n\} = \operatorname{span}(a_1, a_2, \ldots, a_n) = C(A),$$

где $C(A)$ — пространство столбцов матрицы $A$.

**Образ — подпространство.** Если $T$ линейно, то $\operatorname{Im}(T)$ является подпространством $\mathbb{R}^m$. Действительно: для $T(\vec{a}), T(\vec{b}) \in \operatorname{Im}(T)$ имеем $T(\vec{a})+T(\vec{b}) = T(\vec{a}+\vec{b}) \in \operatorname{Im}(T)$, и аналогично $cT(\vec{a}) = T(c\vec{a}) \in \operatorname{Im}(T)$.

Таким образом, вопрос о том, принадлежит ли $\vec{b}$ образу преобразования $T(\vec{x}) = A\vec{x}$, равносилен вопросу: входит ли $\vec{b}$ в пространство столбцов $C(A)$? Если да — система $A\vec{x} = \vec{b}$ совместна, если нет — решений нет.
