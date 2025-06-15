# Мета классы

- Классы, которые создают другие класса, так как все в python - объект
- type(name, bases, dct)
-
    - name - имя
    - bases - базовые классы
    - dct - свойства

```python
class Point:
    MAX_COORD = 100
    MIN_COORD = 0


# добавление метода
def method(self):
    print(self.__dict__)


# вместо method может быть lambda
A = type('Point', (), {'MAX_COORD': 100, 'MIN_COORD': 0, 'method': method})
pt = A()

```

# Пользовательские мета классы

- через функцию

```python
def create_class_point(name, base, attrs):
    attrs.update({'MAX_COORD': 100, 'MIN_COORD': 0})
    return type(name, base, attrs)


# при создании отработает create_class_point
class Point(metaclass=create_class_point):
    # добавит get_coords в него
    def get_coords(self):
        return (0, 0)
```

- через класс

```python
class Meta(type):
    # через new
    def __new__(cls, name, base, attrs):
        attrs.update({'MAX_COORD': 100, 'MIN_COORD': 0})
        return type(name, base, attrs)

    # через new int
    def __init__(cls, name, base, attrs):
        super().__init__(name, base, attrs)
        cls.MAX_COORS = 0
        cls.MIN_COORS = 1000


# при создании отработает create_class_point
class Point(Meta):
    # добавит get_coords в него
    def get_coords(self):
        return (0, 0)
```