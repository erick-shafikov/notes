```python

import random

a, b = 3
# генерация случайных величин от 0 до 1
a = random.random()
# генерация случайных величин float от a до b
random.uniform(a, b)
# int
random.randint(a, b)
# с шагом
random.randrange(-3, 10, 2)

# с нормальным распределения
random.gauss(mu, sigma)
```

```python
import random

some_iter = []
count = 2
# выбор произвольного
a = random.choice(some_iter)
# перемешивание
random.shuffle(some_iter)
# выбор нескольких случайных
random.sample(some_iter, count)
```

```python
import random

number = 1
# зерно последовательности
random.seed(number)
```

<!--  -->

```python

```