# создание функции

```python

def send_mail():
    text = 'text'
    print(text)


# вызов
send_mail()

```

# Создание функции

```python
def some_func(arg_1, arg_2):
    print(arg_1, arg_2)
```

# return

```python
def get_sqrt(x):
    res = None if x < 0 else x ** 5
    return res


# возврат нескольких значений
def get_sqrt_2(x):
    res = None if x < 0 else x ** 5
    return res, x


a, b = get_sqrt(1)
```

```python

PERIMETER = True
if PERIMETER:
    def get_rec(a, b):
        return (a + b) * 2
else:
    def get_rec(a, b):
        return a * b

# if не влияет на область видимости
print(get_rec(1, 3))
```

# help

```python
def func():
    """some description"""


# выведет описание, если оно есть
help(func)

import time

start = time.time()  # получить timestamp
```

# Именованные параметры

```python

def get_v(a, b, c):
    return a * b * c


v = get_v(1, 2, c=3)
```

# Параметры по умолчанию

Существую фактические и формальные параметры

```python
# формальный параметр verbose, фактические a,b,c
def get_v(a, b, c, verbose=True):
    v = a * b * c
    if verbose:
        print(v)

    return a * b * c
```

Формальные параметры и изменяемы типы данных

```python
def func(value, lst=[]):
    lst.append(value)
    return lst


l = func(1)
l = func(2)

print(l)  # [1, 2]
```

# Параметры по умолчанию

kwargs - словарь
args - картеж

```python
# функция принимает произвольное количество аргументов
def func(*args):
    pass


def func1(*args, sep=1, second=True):
    pass


# функция принимает произвольное количество аргументов и именованные
def func2(*args, **kwargs):
    # собрать в kwargs формальные параметры
    param = kwargs['some_formal_param']


def func3(*args, some='', **kwargs):
    pass


# микс, формальные должны идти до **kwargs
def func4(*args, **kwargs):
    # микс, формальные должны идти до **kwargs
    if 'some' in kwargs and kwargs['some']:
        print()

```

# lambda

Анонимные функция

- нужны для использования в конструкции
- должны быть в одну строчку
- делают одну операцию
- нельзя присваивать внутри

```python
s = lambda a, b: a + b
s(1, 2)  # 3
```

```python
lst = [1, 2, 3, 4, 5, 6]


def get_filter(a, filter=None):
    if filter is None:
        return a

    res = []
    for x in a:
        if filter(x):
            res.append(x)

    return res


r = get_filter(lst, lambda x: x > 0)
```

# Область видимости

- если в файле объявлены переменные - они глобальные
- функция создает пространство имен, цикл нет
- к глобальным переменным можно обратиться внутри функции

## global

```python
N = 1000


def some_func():
    # обращение к глобально переменной
    # не должно дублировать
    global N
    # далее мы ее можем использовать

```

## nonlocal

- переменная находится в ближайшей области видимости
- nonlocal нельзя прокидывать на global пространство

```python
x = 3


def outer():
    x = 1

    def inner():
        # обращение x из внешнего
        nonlocal x
        x = 2
        print('x:', x)

    inner()
    print(x)
    outer()


```

# замыкание

- все локальные окружения ссылаются друг на друга
- при каждом вызове say_name будет создаваться новое локальное окружение

```python
def say_name(name):
    def say_goodbye():
        print(f'gb {name}')

    return say_goodbye


f = say_name('x')
f()
```

```python
def counter(start=0):
    def step():
        nonlocal start
        start += 1
        return start

    return step


c1 = counter(10)
c2 = counter()
print(c1(), c2())  # 11 1
print(c1(), c2())  # 12 2
print(c1(), c2())  # 13 3
```

# декораторы

```python

def func_decorator(func):
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return res

    return wrapper


# первый вариант вызов с оберткой
def some_func():
    print()


some_func = func_decorator(some_func)
some_func()


# второй вариант с помощью специального синтаксиса декоратора
@func_decorator
def some_func():
    print()


some_func()
```

## Параметры для декоратора

```python
# для сохранения имени и описания функции обернутой в декоратор
# from functions import wraps
import math


# оболочка для аргумента декоратора
def df_decorator(dx=0.01):
    def func_decorator(func):
        # wraps(func)

        def wrapper(x, *args, **kwargs):
            res = (func(x + dx, *args, **kwargs) - func(x, *args, **kwargs)) / dx

            return res
            # сохранение имени функции (если не используется wraps)
            # wrapper.__name__ = func.__name__

        return wrapper

    return func_decorator


# вызов, можем передать параметры в декоратор
@df_decorator(dx=0.001)
def sin_dx(x):
    return math.sin(x)
```

