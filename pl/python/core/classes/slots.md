```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Points2D:
    # какие поля будут разрешены
    __slots__ = ('x', 'y')
    MAX_CORDS = 100

    def __init__(self, x, y):
        self.x = x
        self.y = y


p = Points2D(1, 2)
print(p.x)  # 1
print(p.y)  # 2
# можно удалять
print(p.__dict__)  # ошибка
p.__sizeof__()  # будет меньше
```

# Наследование

```python

class Points2D:
    # какие поля будут разрешены
    # как коллекция _slots__ = 'x',
    __slots__ = ('x', 'y', '__length')
    MAX_CORDS = 100

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__length = (x * x + y * y) ** 0.5

    # работает с @property
    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self, value):
        self.__length = value


# есть в dict
class Point3D(Points2D):
    # унаследуют __slots__ родителей
    __slots__ = ()


p3 = Point3D(1, 2)
p3.z = 4  # ок, но в словаре __dict__ не будет x, y, только если проинициализировать в __init__
```

<!--  -->

```python

```