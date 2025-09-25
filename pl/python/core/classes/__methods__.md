# init

- вызывается перед созданием экземпляра

```python
class Point:
    color = 'red'
    circle = 2

    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b

    def set_cords(self, x, y):
        self.x = x
        self.y = y

    def get_cords(self):
        return (self.x, self.y)


# если не передать - будет ошибка (если не значений по умолчанию)
pt = Point(1, 2)
```

Использование init:

- установка атрибутов
- работа с мутирующими объектами
- обычная инициализация классов, стандартный флоу

# del

финализатор

```python
class Point:
    color = 'red'
    circle = 2

    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b

    # явно указывать ненужно, этим занимается сборщик мусора
    def __del__(self):
        print('Объект удален')

    def set_cords(self, x, y):
        self.x = x
        self.y = y

    def get_cords(self):
        return (self.x, self.y)


# если не передать - будет ошибка (если не значений по умолчанию)
pt = Point(1, 2)
```

# new

Вызывается после создания класса.

```python
class Point:
    # cls ссылается на класс Point
    # должен возвращать адрес нового объекта
    # аргументы обязательные
    def __new__(cls, *args, **kwargs):
        # super() - базовый класс
        return super().__new__(cls)

    # self ссылает на экземпляр класса
    def __init__(self):
        print()

```

Создание singleton

```python
class DataBase:
    # ссылка на соединение
    __instance = None

    def __new__(cls, *args, **kwargs):
        # при создании объекта
        # если значение __instance == none
        if cls.__instance is None:
            # то создать новый экземпляр
            cls.__instance = super().__new__(cls)
        # в противном случае вернуть __instance
        return cls.__instance

    def __del__(self):
        DataBase.__instance = None

    def __init__(self, user, psw, port):
        self.user = user
        self.psw = psw
        self.port = port

    def connect(self):
        print(f'соединение с БД: {self.user}, {self.psw}, {self.port}')

    def close(self):
        print(f' close {self.user}, {self.psw}, {self.port}')

    def read(self):
        return f' some data {self.user}, {self.psw}, {self.port}'

    def write(self):
        print(f' write {self.user}, {self.psw}, {self.port}')


db = DataBase('user', 'pass', 2000)
db2 = DataBase('user2', 'pass2', 2001)  # db1 == db2
```

PositiveInt пример

```python
class PositiveInt(int):
    def __init__(cls. value:int) -> int:
        if value < 0:
            raise ValueError("must be > 0")
        else:
            return super().__new__(cls, value)
```

Shape пример

```python
from typing import Union, Any

class Shape:
    def __new__(cls, shaper_type: str, *args: Any, **kwargs: Any) -> Union['Circle', 'Rectangle']:
        if shape_type == 'circle':
            return Circle(*args, **kwargs)
        elif shape_type == 'rectangle':
            return Rectangle(*args, **kwargs)
        else:
            raise ValueError('Unknown shape')

class Circle:
    def __init__(self, radius:float) -> None:
        self.radius: float = radius
        self.area: float = 3.14 * radius ** 2

class Rectangle:
    def __init__(self, width:float, height:float) -> None:
        self.width: float = width
        self.height: float = height
        self.area: float = width * height
```

```python
class HTTPStatus(Enum):
    OK = (200, 'OK', True)
    NOT_FOUND = (404, 'Not Found', False)
    INTERNAL_ERROR = (500, 'Internal server error')

    def __new__(cls, code:int, message:str, is_success:bool) -> 'HTTPStatus':
        obj = object.__new__(cls)
        obj._value_ = code
        obj.code = code
        obj.message = message
        obj.is_success = is_success

        return obj
```

- !!! Должен возвращать что-либо отличное от None - ошибка

Использование new:

- singleton
- фабричный метод
- контроль экземпляра
- мета-классы, мета-программирование
- если нужно вернуть другой тип
- для классов унаследованных от не мутирующих классов

# dunder - методы (**getattribute**, **setattr**, **getattr**, **delattr**)

```python
class Point:
    MAX_CORD = 100
    MIN_COORD = 2

    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b

    # при обращении через экземпляр класса
    # применение
    def __getattribute__(self, item):
        if item == 'x':
            raise TypeError('error msg')
        # object - класс от которого наследуются остальные
        # перенаправляет
        return object.__getattribute__(self, item)

    # при присвоении
    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        # альтернативный подход отправит в рекурсию
        # self.x = value
        # выход
        # self.__dict__[key] = value

    # если идет обращение идет к несуществующему атрибуту
    def __getattr__(self, item):
        # предотвратит ошибку при обращении к несуществующему атрибуту
        return False

    # при удалении
    def __delattr__(self, item):
        object.__delattr__(self, item)


```

# **call**

Вызывается при создании экземпляра класса именно он позволяет вызывать класс со скобками.
Такие классы называются функторы

```python

class Counter:
    def __init__(self):
        self.__counter = 0

    #  поведение по умолчанию
    # def __call__(self, *args, **kwargs):
    #    obj = self.__new__(self, *args, **kwargs)
    #    self.__init__(obj, *args, **kwargs)
    #    return obj

    # для передачи аргументов
    def __call__(self, step=1, *args, **kwargs):
        print('call')
        self.__counter += step
        return self.__counter


с = Counter()
# увеличит значение __counter
с(1)
с(2)
res = с(3)

с2 = Counter()  # независимый
с2()
```

пример

в качестве замыкания

```python
class StripChars:
    def __init__(self, chars):
        self.__counter = 0
        self.__chars = chars

    def __call__(self, *args, **kwargs):
        if not isinstance(args[0], str):
            raise TypeError('не строка')

        return args[0].strip(self.__chars)


s1 = StripChars("?!:.;")
res = s1(" Hello world! ")  # удалены ! и пробелы
```

в виде декоратора

```python
import math


class Derivate:
    def __init__(self, func):
        self.__fn = func

    def __call__(self, x, dx=0.0001, *args, **kwargs):
        return (self.__fn(x + dx) - self.__fn(x)) / dx


def df_sin(x):
    return math.sin(x)


# теперь функция ссылается на класс
df_sin = Derivate(df_sin)
```

# **str**()

Отображает информацию об объекте

```python
class Cat:
    def __init__(self, name):
        self.name = name

    # отобразится в print
    # отобразиться в str(instance)
    def __str__(self):
        return f'{self.name}'
```

# **repr**()

Как будет отображаться во время отладки экземпляры класса

```python
class Cat:
    def __init__(self, name):
        self.name = name

    # если просто ввести cat в консоли
    def __repr__(self):
        return f'{self.__class__}: {self.name}'
```

# **len**()

Позволит применять функцию len к экземплярам

```python
class Point:
    def __init__(self, *args):
        self.__cords = args

    def __len__(self):
        return len(self.__cords)


p = Point(1, 2, 3)
len(p)  # 3

```

Позволит применять функцию abs к экземплярам

# **abs**()

```python
class Point:
    def __init__(self, *args):
        self.__cords = args

    def __abs__(self):
        return list(map(abs, self.__cords))


p = Point(1, -2, -3)
abs(p)  # [1,2,3]
```

# Математические операции

## **add**()

```python
class Clock:
    __DAY = 86400

    def __init__(self, seconds: int):
        self.seconds = seconds % self.__DAY

    def get_time(self):
        s = self.seconds % 60
        m = (self.seconds // 60) % 60
        h = (self.seconds // 3600) % 24
        return f'{self.__get_formated(h)}:{self.__get_formated(m)}:{self.__get_formated(s)}'

    @classmethod
    def __get_formated(cls, x):
        return str(x).rjust(2, '0')

    # метод будет возвращать новый экземпляр с измененными параметрами
    def __add__(self, other):
        sc = None
        if type(other) == Clock:
            sc = other.seconds

        if type(other) == int:
            sc = other

        return Clock(self.seconds + sc)

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        sc = other

        if isinstance(other, Clock):
            sc = other.seconds

        self.seconds += sc
        return self


c1 = Clock(1000)
c2 = Clock(2000)

c3 = c1 + c2
c4 = c1 + 100
# Ошибка лечится radd
c5 = 100 + c1
c1 += 100
```

- **sub**() - для вычитания,
- **mul**() - для умножения,
- **truediv**() - для деления
- **floordiv**() - x // y
- **mod**() - x % y

# методы сравнения

- **eq**() - равенство ==
- **ne**() - != вызывает **eq**() если неопределённа и значение инвертируется
- **lt**() - <
- **le**() - <=
- **gt**() - > - взывает **lt**() если неопределённа и инвертирует результат
- **ge**() - >=

```python
class Clock:
    __DAY = 86400

    def __init__(self, seconds: int):
        self.seconds = seconds % self.__DAY

    def __eq__(self, other):
        sc = self.__verify_data(other)
        return self.seconds == sc

    def __lt__(self, other):
        sc = self.__verify_data(other)
        return self.seconds < sc

    def __le__(self, other):
        sc = self.__verify_data(other)
        return self.seconds <= sc

    # вспомогательный метод
    @classmethod
    def __verify_data(cls, other):
        if not isinstance(other, (int, Clock)):
            raise TypeError('type error')

        return other if isinstance(other, int) else other.seconds


c1 = Clock(100)
c2 = Clock(100)

print(c1 == c2)
```

# hash

- для вычисления hash функции, может подменить

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # для этого класса hash работать не будет
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))
```

# bool

- преобразование к логическому типу
- если этого метода нет, то используется **len**()
- должен возвращать bool

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x * self.x + self.y * self.y

    def __bool__(self):
        return self.x == self.y


a = Point(3, 4)

if a:
    print('true')
```

# **getitem**(self, item), **setitem**(self, key, value), **delitem**(self, key)

- позволяет обращаться по индексу к экземпляру класса

```python
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    # позволит обращаться
    def __getitem__(self, item):
        if 0 <= item < len(self.marks):
            return self.marks[item]

    # позволит изменять
    def __setitem__(self, key, value):
        if not isinstance(key, int) or key < 0:
            raise TypeError('error')
        # позволит расширить
        if key >= len(self.marks):
            off = key + 1 - len(self.marks)
            self.marks.extend([None] * off)

        self.marks[key] = value

    def __delitem__(self, key):
        if not isinstance(key, int) or key < 0:
            raise TypeError('error')

        del self.marks[key]


s1 = Student('Max', [5, 5, 3, 2, 5])
print(s1[2])  # 3
s1[2] = 4
print(s1[2])  # 4
del s1[2]
```

# **iter** и **next**

- позволит перебирать объекта

```python
class Frange:
    def __init__(self, start=0.0, stop=0.0, step=1.0):
        self.start = start
        self.stop = stop
        self.step = step
        # для первого next() чтобы начинался с первого
        self.value = self.start - self.step

    def __next__(self):
        if self.value + self.step < self.stop:
            self.value += self.step
            return self.value
        else:
            raise StopIteration

    # обнуляется для первого шага, так как методы перебора используют iter
    def __iter__(self):
        self.value = self.start - self.step
        return self


fr = Frange(0, 2, 0.5)
print(next(fr))
print(next(fr))
print(next(fr))
print(next(fr))

for x in fr:
    print(x)


# пример вложенного
class Frange2D:
    def __init__(self, start=0.0, stop=0.0, step=1.0, rows=5):
        self.start = start
        self.stop = stop
        self.step = step
        self.rows = rows
        self.fr = Frange2D(start, stop, step)

    def __iter__(self):
        self.value = 0
        return self

    def __next__(self):
        if self.value < self.rows:
            self.value += 1
            return iter(self.fr)
        else:
            raise StopIteration


fr = Frange2D(0, 2, 0.5, 4)

for row in fr:
    for x in row:
        print(x)
```

##########################################

```python

```
