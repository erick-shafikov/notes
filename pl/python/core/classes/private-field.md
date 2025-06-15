from xml.dom import ValidationErr_private - можно обращаться из класса и дочерних экземпляров, только договренность

__protected - можно обращаться только внутри класса

```python
# _private
# __protected
class Point:
    color = 'red'
    circle = 2

    def __init__(self):
        self.__y = None

    def set_coords(self, x, y):
        self._x = x
        self.__y = y

    def get_cords(self):
        return (self.__y)

    def set_cords(self, y):
        self.__y = y

    def set_cords_condition(self, y):
        # доп проверка
        if y < 0:
            raise TypeError('error message')

        self.__y = y

    # приватное свойство
    @classmethod
    def __check_value(cls, x):
        return type(x) in (int, float)


a = Point()
a._x  # можно обращаться
a.__x  # ошибка
a.get_cords()
a.set_cords(1)
# обход
print(a._Point__y)
```

Модуль accessify

```python
from accessify import private, protected


class Class:
    @private
    def method(self):
        pass

    @protected
    def method(self):
        pass

```

<!--  -->

```python

```