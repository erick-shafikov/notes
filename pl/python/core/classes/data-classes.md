# Data classes

```python
# есть класс, который хранит некоторые данные
# много boilerplate
class Thing:
    # dims=[] ссылается на список
    def __init__(self, name, weight, price=20, dims=[]):
        self.name = name
        self.weight = weight
        self.price = price

    def __repr__(self):
        return f"thing {self.__dict__}"

```

```python
from dataclasses import dataclass, field


@dataclass
# порядок имеет значение
class ThingData:
    name: str
    weight: int
    # нельзя добавлять изменяемы данные
    # default_factory для инициализации
    dims: list = field(default_factory=list)
    price: float = 20


#     Можно прописать все __методы__

td = ThingData('x', 100, [101])
td_2 = ThingData('x', 101, [102])
td_3 = ThingData('x', 101, [102])

print(td == td_2)  # False
print(td_2 == td_3)  # True так как пере определится __eq__
```

# Вычисляемые свойства

```python
from dataclasses import field, InitVar, dataclass


class Vector3D:
    # clac_len флаг для подсчета свойства
    def __init__(self, x: int, y: int, z: int, clac_len=True):
        self.x = x
        self.y = y
        self.z = z
        # вычисляемое свойство
        self.length = (x * x + y * y + z * z) ** 0.5 if clac_len else 0


@dataclass
class V3D:
    x: int = field(repr=False)
    y: int = field(compare=False)
    z: int
    # что бы добавить в __repr__
    # init=False не будет требовать
    length: float = field(init=False, default=0)
    # InitVar отправит инициализацию calc_len в __post_init__
    calc_len: InitVar[bool] = True

    # здесь вычисляемые свойства не могут быть определены

    # метод __post_init__ вызывается после инициализации статических параметров 
    def __post_init__(self, calc_len: bool):
        # в repr не передавать
        if calc_len:
            self.length = (self.x * self.x + self.y * self.y + self.z * self.z) ** 0.5


```

- field имеет параметры:

- repr (True) - использовать ли параметр в __repr__
- compare (True) - использовать ли при сравнении
- default - значение по умолчанию

# Параметры @dataclass

- init (True) - вызывать или нет инициализатор (без __init__) для базовых классов
- repr (True) - формировать ли repr
- eq (True) - формировать ли eq
- order (False) - сравнения объектов, добавится < > <= >=, нельзя переопределять методы сравнения
- unsafe_hash () - влиять на формирование hash
- frozen (False) - атрибуты будут неизменяемые
- slots - для слотов

# наследование

```python
from dataclasses import field, InitVar, dataclass
from typing import Any


# вспомогательный класс
class GoodsMethodFactory:
    @staticmethod
    def get_init_measure():
        return [0, 0, 0]


@dataclass
class Goods:
    # Инициализатор
    # def __init__( uid: Any, price: Any, wight: Any)
    current_uid = 0  # так как нет аннотации типов, то атрибут не будет добавлен
    uid: int = field(init=False)
    # если добавить значение по умолчанию, то в дочерних классах возникнет ошибка
    price: Any = None
    wight: Any = None

    def __post_init__(self):
        Goods.current_uid += 1
        self.uid = Goods.current_uid


@dataclass
class Book(Goods):
    # Инициализатор
    # новые атрибуты в конец списка атрибутов
    # def __init__( uid: Any, price: float, wight: int | float, title: str, author:str)
    title: str = ''
    author: str = ''
    price: float = 0
    weight: int | float = 0
    measure: list = field(default_factory=GoodsMethodFactory.get_init_measure)

    def __post_init__(self):
        # по умолчанию базовый класс пере определится
        super().__post_init__()
```

# make_dataclass

альтернативный вариант создания data classes

make_dataclass(cls_name, fields, *, bases=(), namespace=None, init=True)

- cls_name - название
- fields - поля
- \*- позиционные аргументы
- bases - базовые классы

```python
from dataclasses import field, InitVar, dataclass, make_dataclass


class Car:
    def __init__(self, model, max_speed, price):
        self.model = model
        self.max_speed = max_speed
        self.price = price

    def get_max_speed(self):
        return self.max_speed


CarData = make_dataclass("CarData", [("model", str),
                                     "max_speed",
                                     ("price", field(default=0))],
                         namespaces={'get_max_speed': lambda self: self.max_speed})
```

