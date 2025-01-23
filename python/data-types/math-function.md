```python
# модуль
# обязательно с числовым аргументом
abs(1.5)
# min - минимальное и максимальное
# обязательно с числовым аргументом или одним аргументом
min(1, 2, 3, 4, 5) #1
max(1, 2, 3, 4, 5)
pow(6, 2) == 6 ** 2
round(0.5) # 0 иногда в большую сторону
round(1.5) # 2 иногда в меньшую
round(1.001, 2) # до сотых
round(7.8, -1) # до десятых, -2 до сотен
#

# Комбинации
max(1, 2, abs(-3))
```

Модули

```python
import math

math.ceil(5.2) #6 наибольшее целое
math.floor(5.99) #5 наименьшее целое
math.factorial(3) #6
# отбросить целую часть
math.trunc(5.8) #5
int(5.8)
# логарифмы
math.log2(4)
math.log10(100)
math.log(3) # loge
math.log(27,3)
math.sqrt(49)
math.sin()
math.cos()
math.pi
math.e

from math import factorial as f
```
