# np.linalg.inv

свойства:

- матрица должна быть квадратной и невырожденной (`det(A) != 0`), иначе `LinAlgError: Singular matrix`
- результат всегда `float64`, проверять через `np.allclose`, не через `==`
- для решения системы `Ax = b` лучше `np.linalg.solve(A, b)` — быстрее и численно стабильнее
- плохо обусловленные матрицы дают неточный результат без ошибки, проверить: `np.linalg.cond(A)` >> 1e10 — опасно
- поддерживает батч: вход `(batch, n, n)` → выход `(batch, n, n)`

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

A_inv = np.linalg.inv(A)
# array([[-2. ,  1. ],
#        [ 1.5, -0.5]])

# проверка: A @ A_inv == I
np.allclose(A @ A_inv, np.eye(2))  # True
```
