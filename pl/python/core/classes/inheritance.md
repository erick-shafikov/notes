Наследование

```python
class Geom:
    name = 'Geom'

    def __init__(self):
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None

    def set_cords(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class Line(Geom):
    # переопределяет
    name = 'line'

    def draw(self):
        print('draw Line')


class Rect(Geom):

    def draw(self):
        print('draw Rect')


l = Line()

print(l.name)
```

# базовый класс object

- все наследуются от object

```python

class Geom(object):
    pass

```

# issubclass и isinstance

- issubclass проверка класса только наследование классов, но не объектов
- isinstance проверяет на наследование и объекта и классы

Пример расширения стандартных классов

- все типы - это классы

```python
# расширяем list
class Vector(list):
    def __str__(self):
        # в self пойдет класс list
        return " ".join(map(str, self))
```

# super()

- нужно для делегирования

```python
class Geom:
    name = 'Geom'

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        print('init Geom')

    def draw(self):
        print('draw')


class Line(Geom):
    def __init__(self, x1, y1, x2, y2):
        # дублирование
        # self.x1 = x1
        # self.y1 = y1
        # self.x2 = x2
        # self.y2 = y2
        super().__init__(x1, y1, x2, y2)
        print('init Geom')

    def draw(self):
        print('draw Line')


# в инициализаторе добавляется поле fill
# делегирование
class Rect(Geom):
    def __init__(self, x1, y1, x2, y2, fill=None):
        # вариант плохой
        # Geom.__init__(self, x1, y1, x2, y2)

        # присвоить x1, y1, x2, y2
        # должен быть первым
        super().__init__(x1, y1, x2, y2)
        # дублирование
        # self.x1 = x1
        # self.y1 = y1
        # self.x2 = x2
        # self.y2 = y2
        # дополнительное поле
        self.fill = fill
        print('init Geom')

    def draw(self):
        print('draw Rect')


l = Line(1, 2, 3, 4)
# вызывается метод __call__ в Line
# __call()__(self, *arg, **kwarg): <- __call__ берется из мета класса
#     obj = self.__new__(self, *arg, **kwarg) <- __new__ берется из object
#     self.__init__(obj, *arg, **kwarg) <- __init__ берется из Geom если нет в Line
#     return obj
```

# private и protected поля

- private поля не доступны в методах суб-классов, т.е. только для класса, распространяется на методы
- protected будут доступны в суб-классах, нижнее подчеркивание - договоренность

```python
class Geom:
    __name = 'Geom'

    def __init__(self, x1, y1, x2, y2):
        self.__x1 = x1
        self.__y1 = y1
        self.__x2 = x2
        self.__y2 = y2
        print('init Geom')

    # будет доступно только внутри Geom при наследовании
    def __verify_cord(self, cord):
        pass

    # будет доступно в суб-классах
    def _other_method(self):
        pass

    # будет доступно в суб-классах так как
    def __some_dunder_method__(self):
        pass


class Rect(Geom):
    def __init__(self, x1, y1, x2, y2, fill=None):
        super().__init__(x1, y1, x2, y2)
        self.__fill = fill


rect = Rect(1, 2, 3, 4)
# '_Geom__x1': 1, '_Geom__y1': 2,... _Rect__fill'
print(rect.__dict__)
```

# Множественное наследование

```python
class Goods:
    def __init__(self, name, weight, price, mixin_params):
        # заставит обратиться к инициализатору MixinLog
        super().__init__(mixin_params)
        self.name = name
        self.weight = weight
        self.price = price

    def print_info(self):
        print(f'{self.name}')


# миксин может быть только с self
class MixinLog:
    ID = 0

    # нежелательно использовать какие-либо mixin_params
    def __init__(self, mixin_params):
        MixinLog.ID += 1
        self.id = MixinLog.ID
        self.mixin_params = mixin_params

    def save_sale_log(self):
        print(self.id)

    def print_info(self):
        print(f'{self.id}')


# множественное наследование,
# но MixinLog не запустится init без super()
# порядок имеет значение
class NoteBook(Goods, MixinLog):
    def print_info(self):
        # если нужно вызвать метод именно миксина
        MixinLog.print_info(self)


# MRO - method resolution order позволяет проходиться по всей цепочки объектов
# позволит отследить как идет вызовы
print(NoteBook.__mro__)

```

# Наследование от базовых классов

- при наследовании от неизменяемых типов данных должны переопределять __new()__
- У неизменяемых объектов (tuple, str, int) нужно переопределять __new__, а не только __init__.
- __new__ отвечает за создание объекта, а __init__ — за дополнительную инициализацию.

```python
class TupleLimit(tuple):
    def __new__(cls, iterable, max_length=0):
        if len(iterable) > max_length:
            raise ValueError("число элементов коллекции превышает заданный предел")
        return super().__new__(cls, iterable)

    def __init__(self, iterable, max_length=0):
        self.max_length = max_length

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return self.__str__()
```

<!--  -->

```python

```