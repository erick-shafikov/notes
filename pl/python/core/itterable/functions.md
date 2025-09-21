# any all

```python
a = [0, 1, 2.5, 'string']
all(a)  # False превратятся в булево значение все должны быть True
any(a)  # True хоть один должен быть True
```

# filter

```python
a = [1, 2, 3, 4, 5]

b = filter(lambda x: x % 2 == 0, a)

```

# map

Для каждого вызывается функция, возвращает перебираемый объект

- функция должна принимать один аргумент
- дважды нельзя пройти

```python
some_iterable = []


def fun(x):
    pass


b = map(fun, some_iterable)
b = (fun(x) for x in some_iterable)
b = (lambda s: print(s), some_iterable)
```

# sorted

```python
a = [3, 4, 1, 7]
# изменить текущий, ⇒ None
a.sort()
# изменить текущий, ⇒ новый список
b = sorted(a, reverse=True)
```

## key

Позволяет отсортировать итерируемые

```python
def is_odd(x):
    return x % 2


a = [4, 3, -10, 1, 7, 12]
# первый элементы будут четные (дял которых is_odd => true)
b = sorted(a, key=is_odd)  # [4,10,12,3,1,7]

```

```python
a = []
sorted(a, key=lambda x: x[-1])
```

# zip

возвращает объединение в картеж перебираемых объектов
в каждый картеж будут добавлены значение

```python

a = [1, 2, 3, 4]
b = [5, 6, 7, 8, 9, 10]

z = zip(a, b)

for x in z:
    print(x)

# (1, 5)
# (2, 6)
# (3, 7)
# (4, 8)
```

```python
a = [1, 2, 3, 4]
b = [5, 6, 7, 8, 9, 10]
c = 'python'

z = zip(a, b, c)
t1, t2, t3 = zip(*z)
print(t1, t2, t3)

# (1, 2, 3, 4)
# (5, 6, 7, 8)
# (p, y, t, h)
```
