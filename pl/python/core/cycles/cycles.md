# while

```python
N = 1000
s = 0
i = 0

while i <= N:
    s += i
    i += 1

# 
pass_true = 'password'
ps = ''

while ps != pass_true:
    ps = input('Введите пароль: ')

print('Вход')
```

# break

```python
while True:
    i += 1
    # прерывание после первой итерации
    break
```

```python
# перебрать только четные
numbers = [1, 2, 3, 4]

while i < len(numbers):
    flFind = d[i] % 2 == 0
    if flFind:
        break
    i += 1

```

# continue

переход на следующий цикл

```python
s = 0
d = 1

while d != 0:
    d = int(input('Ведите четное значениеЖ'))
    if d % 2 == 0:
        continue

    s += d

```

# while else

завершение в штатном режиме

```python
# s = 1/2 + 1/3 + 1/4

s = 0
i = -10

while i < 100:
    if s == 0:
        break
    s += 1 / i
    i += 1
else:
    # если дошли до конца условия
    print('Завершено без break')
```

# for

```python
numbers = [1, 2, 3, 4]

for x in numbers:
    print(x)

# не будет менять
for x in numbers:
    x = 3
```

# Вложенные циклы

```python

for i in range(1, 4):
    for j in range(1, 6):
        pass
# код операций

```

<!--------------------------------- -->

```python

```