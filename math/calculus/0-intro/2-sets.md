## Множества и декартово произведение

### Множество

**Множество** — совокупность различимых объектов (элементов). Запись $a \in A$ означает, что $a$ — элемент множества $A$; $a \notin A$ — не элемент.

Основные операции:
- $A \cup B$ — объединение: $\{x \mid x \in A \text{ или } x \in B\}$
- $A \cap B$ — пересечение: $\{x \mid x \in A \text{ и } x \in B\}$
- $A \setminus B$ — разность: $\{x \mid x \in A \text{ и } x \notin B\}$
- $A \subset B$ — включение: каждый элемент $A$ есть элемент $B$

Законы де Моргана:
$$\overline{A \cup B} = \overline{A} \cap \overline{B}, \qquad \overline{A \cap B} = \overline{A} \cup \overline{B}$$

### Декартово произведение

**Декартово произведение** $A \times B$ — множество всех упорядоченных пар:

$$A \times B = \{(a, b) \mid a \in A,\; b \in B\}$$

Упорядоченная пара $(a, b) = (c, d)$ тогда и только тогда, когда $a = c$ и $b = d$.

Пример: $[0,1] \times [0,1]$ — единичный квадрат в $\mathbb{R}^2$.

Обобщение: $A_1 \times A_2 \times \cdots \times A_n$ — множество $n$-ок $(a_1, \ldots, a_n)$. В частности, $\mathbb{R}^n = \mathbb{R} \times \cdots \times \mathbb{R}$.

### Соответствие

**Соответствие** между множествами $A$ и $B$ — любое подмножество $G \subset A \times B$. Если $(a, b) \in G$, говорят, что $a$ соответствует $b$.

Область определения: $\mathrm{Dom}(G) = \{a \in A \mid \exists b: (a,b) \in G\}$.  
Область значений: $\mathrm{Im}(G) = \{b \in B \mid \exists a: (a,b) \in G\}$.
