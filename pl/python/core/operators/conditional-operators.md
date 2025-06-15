# if

```python
if x > 0:
  # do something
```

```python
if x > 0:
  # do something
else: 
  # do another
```

```python
if x > 0:
  if x < 2:
    # do_do_nested_something_something
  else:
    # do_nested_something
else: 
  # do another
```

# elif

```python
if condition:
    # do something
elif condition:
    # do something 1
elif condition:
    # do something 2
else:
   # do something 2
```

# тернарный оператор

```python
# вернуть а, если a > b, иначе b
res = a if a > b else b
# можно выполнять 1 операцию
res = a - 2 if a > b else b - 3
# в листах
[1, 2, a if a > b else b]
# в строках
"a - " + ('четное' if a % 2 == 0 else "нечетное") + "число"
# вложенные
a = 2
b = 3
c = -4

d = (a if a > c else c ) if a > b else (b if b > c else c)
```
