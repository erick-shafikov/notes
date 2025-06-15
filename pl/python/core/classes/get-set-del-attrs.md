- атрибуты общие
- имя ищутся от объекта класса

```python
class Point:
    # атрибуты, свойства класса
    """Описание класса"""
    color = 'red'
    circle = 2


Point.color = 'black'
# все свойства в __dict__
print(Point.__dict__)
# описание
print(Point.__doc__)

# создание объекта
# пространство имен будет пустое {}
# свойства color и circle не существуют отдельно от Point
a = Point()
# изменим 
Point.circle = 1
# изменилось в экз
a.circle == 1  # True
# но если создать в самом экземпляре
a.circle = 2  # атрибут circle изменится только в а, оно появится как атрибут a. До этого не существовал в a

# Добавление
Point.type_at = 'disc'
# или с помощью setattr динамически меняются или добавляются
# Point.prop = 1
setattr(Point, 'prop', '1')

# getattr позволяет читать  если нет, то будет ошибка
getattr(Point, 'some_attr')

# удалить
del Point.prop
delattr(Point, 'type_pt')

# проверка hasattr проверит и в классе и в объекте
hasattr(Point, 'circle')  # true
# проверка в объекте локальных свойств
print('job' in a.__dict__)

# проверка на заимствования
print(type(a) == Point)  # True
print(isinstance(a, Point))  # True

a.x = 1
```

<!--  -->

```python

```