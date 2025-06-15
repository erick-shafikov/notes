- хотя бы 1 блок case

```python
match x:
    case 'x':
        print('x')
    case 'y':
        print('y')
    case _:  # wildcard
        print('other')
```

# Несколько соответствий

```python
match x:
    case 'x' | 'y':
        print('x or y')
    case _:  # wildcard
        print('other')
```

# Дефолтный случай

```python
match x:
    case some_var:  # some_var == x
        # это будет вызываться всегда
        print(some_var)
        # игнорируется
    case 'y':
        print('y')
    case _:  #wildcard
        print('other')

match x:
    # не игнорируется
    case 'y':
        print('y')
    case some_var:  # some_var == x
        # это будет вызываться всегда
        print('x')
    case _:  #wildcard
```

# проверка на тип

```python
match x:
    # если x - строка
    case str():
        print('x')
    case str() as var:  # ссылка на переменную x в случает если x - строка
        # это будет вызываться всегда
        print(var)
        # !!! помнить что bool является подклассом int
    case str(var):  # аналогично
        print(var)
        # проверка на два типа
    case int() | float()
        print(var)
    case _:  # wildcard
```

# с доп условием

```python
match x:
    # доп условие с if
    case int() as var if var > 0:
        print('x')
    case 'y':
        print('y')
    case _:  #wildcard
        print('other')
```

# работа с list и tuple

```python
cmd = ('x', 'y', 'z')
cmd = [1, a, b, c]

match cmd:
    # проверка на не примитивный тип
    case tuple() as book
        print(book)
    # распаковка сработает если три элемента
    case x, y, z:
        print(x, y, z)
    # распаковка более трех
    case x, y, z, *_:
        print(x, y, z)
    # распаковка более трех c доп условиями
    case [x, y, z, *_] if some_condition:
        print(x, y, z)
    # распаковка более трех c доп условиями и проверкой типов
    case [str() as x, int() | float() as y, z, *_] if some_condition:
        print(x, y, z)
    # проверка на разные типы структур данных (имена должны совпадать)
    case (x, y, z) | [_, x, y, z]Ж
        print(x, y, z)
    case _:
        print('неизвестный тип')
```

# работа с dict и set

- для словарей только наличие полей

```python
request = {url: '', method: 'get', timeout: 1000}

match request:
    case {'url': url, 'method': method}:
        print(url, method)
    # проверка типов
    case {'url': str() as url, 'timeout': 1000 | 2000}:
        print(url, method)
    # Доп условия
    case {'url': url, 'timeout': 1000} if len(request) > 3:
        print(url, method)
    # Доп условия для остаточных параметров
    case {'url': url, 'timeout': 1000, **kwargs} if len(kwargs) > 3:
        print(url, method)
    case {'url': url, 'timeout': 1000, **kwargs} if not kwargs:
        print(url, method)
    case _:
        print('error')
```

```python
primary_keys = {1, 2, 3}

match primary_keys:
    case set() as keys if len(primary_keys) == 3:
        print(keys)
    case _:
        print('error')

```

# pass

```python
def connection_db(connect) -> str:
    match connect:
        case {"server": host, 'login': login, 'password': psw, 'port': port}:
            pass
        case {"server": host, 'login': login, 'password': psw, }:
            port = 22
        case _:
            return 'error'

        #host, login, psw, port видны
    return f'connection {host}@{login}.{psw}:{port}'

```

<!--  -->

```python

```