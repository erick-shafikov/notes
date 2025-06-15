Два типа дескрипторов:

- non-data descriptor - только для считывания, имеет тот же приоритет доступа, что и атрибуты класса
- data descriptor - приоритет выше

```python
# проблема дублирования 
class point3D:
    def __init__(self, x, y, z):
        self.__x = x
        self.__y = y
        self.__z = z

    @classmethod
    def verify_cord(cls, coord):
        if type(coord) != int:
            raise TypeError('not a number')

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, cord):
        self.verify_cord(cord)

    # дублирование
    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, cord):
        self.verify_cord(cord)

    # дублирование
    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, cord):
        self.verify_cord(cord)
```

с помощью дескрипторов

```python
# non-data descriptor
class ReadIntX:
    # читает локальное свойство x
    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)


# data descriptor
# создает объект вида {_name: value}
class Integer:
    @classmethod
    def verify_cord(cls, coord):
        if type(coord) != int:
            raise TypeError('not a number')

    # self == ссылка на экз. Integer, owner == ссылка на экз. point3D
    # создаст self.name == '_name'
    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, instance, owner):
        # return instance.__dict__[self.name]
        return getattr(instance, self.name)

    # сработает в момент присваивания
    def __set__(self, instance, value):
        print(f"__set__:{self.name} = {value}")
        # instance.__dict__[self.name] = value
        self.verify_cord(value)
        setattr(instance, self.name, value)

    # есть еще del
    def __delete__(self, instance):
        # Вызывается при удалении атрибута
        # instance — экземпляр, где атрибут удаляется
        delattr(instance, self.name)


class Point3D:
    x = Integer()
    y = Integer()
    z = Integer()

    xr = ReadIntX()

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


p = Point3D(1, 2, 3)

print(p.x)
p.x = 1

# xr = 5 если задать
p.xr = 5
```

- для приватных атрибутов

```python
class Property:
    def __set_name__(self, owner, name):
        self.name = f'_{owner.__name__}__{name}'

    def __get__(self, instance, owner):
        if instance is None:
            return property()
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        setattr(instance, self.name, value)
```