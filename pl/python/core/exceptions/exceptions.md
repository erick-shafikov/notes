# exceptions

- исключения в момент исполнения
- исключения при компиляции

```python
try:
    f = open('file.txt')
#     один тип ошибки
except ValueError:
    print('x')
except (ValueError, ZeroDivisionError):
    print('x')

print('x')
```

Иерархия классов ошибок (класс выше отлавливает класс ниже по иерархии):

Сначала обрабатываются более специфичные классы, так как могут быть отловлены родительскими

# as finally else

## as

Обращение к объекту ошибки

```python
try:
    x, y = map(int, input().split())
    res = x / y
except ZeroDivisionError as z:
    print(z)
except ValueError as z:
    print(z)
```

## else

Штатное выполнение

```python
try:
    x, y = map(int, input().split())
    res = x / y
except ZeroDivisionError as z:
    print(z)
except ValueError as z:
    print(z)
# если ошибок не возникает
else:
    print('ok')
```

## finally

Выполняет всегда

```python
try:
    x, y = map(int, input().split())
    res = x / y
except ZeroDivisionError as z:
    print(z)
except ValueError as z:
    print(z)
# если ошибок не возникает
else:
    print('ok')
finally:
    print('Выполняется в любом случае и ошибки и успеха')
```

Использование - закрытия менеджера контекста файла

```python

try:
    file = open('text.txt')
    s = file.readLines()
finally:
    file.close()
```

- finally выполняется до return
- return выполняет в последнюю очередб

```python


def get_value():
    try:
        x, y = map(int, input().split())
        return x, y
    except ValueError as z:
        return 0, 0
    finally:
        print('выполняются до return')
```

# except propagation

- ошибки определяются в стеке вызова функции
- обработку ошибок выносят на верхний уровень, а пробрасывают ошибки ниже по уровню

стек вызовов:
main -> func1() -> func2

```python
def func2():
    return 1 / 2  # строка 2


def func1():
    return func1()  # строка 5


print('1')
print('2')
print('3')
# при отображении ошибки будут строки 2 и строка 5, 10
func1()  # строка 10
print('4')
print('5')
print('6')
print('7')
print('8')
```

- если добавить обработчики

```python
def func2():
    return 1 / 2  # строка 2


def func1():
    try:
        return func1()  # можно отловить здесь
    except:
        print('error')


print('1')
print('2')
print('3')
# можно отловить здесь
try:
    func1()
except:
    print('error')
print('4')
print('5')
print('6')
print('7')
print('8')
```

# raise

```python
# можно генерировать ошибку
e = ZeroDivisionError('деление не ноль')
raise e


```

Базовый класс - Exception.
Таким образом можно создавать иерархию

```python
class CustomError(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        return f'Ошибка: {self.message}'
```

<!--------------------->

```python

```