# генераторы

синтаксис

```python
[x ** 2 for x in range(5)] # [1, 2, 4, 8, 16]
[1 for x in range(5)] # [1, 1, 1, 1, 1]
[0.5 * x + 1 for range(4)] # [1.0, 1.5, 2.0, 2.5, 3.0]

```

```python
a = [int(d) for d in input.split()] # строку ввода разбить и преобразовать в числа

[x for x in 'python'] # [p, y, t, h, o, n]
```

# с условиями

```python

[x for x in range(-5, 5) if x < 0] # [-5, -4, -3, -2, -1]
[x for x in range(-6, 7) if x % 2 == 0 and x % 3 == 0] # [-6, 0, 6]
```

с тернарным

```python
d = [] # числа

['odd' if x % 2 == 0 else 'even' for x in d]# [even, odd...]
# в несколько строк

['odd' if x % 2 == 0 else 'even'
  for x in d
  if x > 0]
```

# вложенные

```python
a =[(i, j) 
    for i in range(3) 
    for j in range(4)
    ] # [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]

matrix = [[1,2,3], [4,5,6], [7,8,9]]
a = [x 
    for row in matrix
    for x in row
    ] # [1,2,3,4,5,6,7,8,9]

```

```python
M, N = 3, 4

matrix = [[a for a in range(M)] for b in range(N)] # [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]

```


```python
# возвести в квадрат каждый вложенный

A = [[1,2,3],[4,5,6],[7,8,9]]
#      x -------> x in row ---> row in A
AA = [[x ** 2 for x in row] for row in A]#[[1, 4, 9], [16, 25, 36], [49, 64, 81]]

# возвести в квадрат каждый вложенный и распрямить
a = [
  x ** 2
  for row in A
  for x in row
] # [1, 4, 9, 16, 25, 36, 49, 64, 81]

# Транспонирует
A = [[1,2,3,4],[5,6,7,8], [9,0,1,2], [3,4,5,6]]
A = [[row[i] for row in A] for i in range(len(A[0]))]

# вложенные функции
g=[u**2 for u in [x+1 for x in range(5)]]
# g(u(x+1)) = (x+1)^2
```

```python
# разложить строку в словарь
dict([row.split('=') for row in lst_in])
```

# для словарей и множеств

плюс - скорость

```python
# множество
a = {x**2 for x in range(1,5)}
# словарь
a = {x :x**2 for x in range(1,5)}
```

```python
# преобразование к числу
d = [1,2 ,3 '-4', 5, '6']
a = {int(x) for x in d}
# только положительные
a ={int(x) for x in d if int(x) > 0}
# перебор словарей
m = {'one': 1, 'two': 2, 'three': 3}
a = {key.upper(): int(value) for key, value in m.items()}
```

<!-- ----------------------- -->

```python

```