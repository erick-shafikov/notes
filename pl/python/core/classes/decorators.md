# @classmethod

Метод, кготорый имеет ссылку на класс, но не может использоваться в его экземплярах. Нужен для использования констант
класса (пример)

```python

class Vector:
    MIN_COORD = 0
    MAX_COORD = 100

    # декоратор позволяет обращаться к полям класса
    # cls - аргумент ссылка на класс
    # нельзя использовать в экземплярах
    @classmethod
    def validate(cls, arg):
        return cls.MIN_COORD <= arg <= cls.MAX_COORD

    def __init__(self, x, y):
        if self.validate(x) and self.validate(y):
            self.x = x
            self.y = y

    def get_cords(self):
        return self.x, self.y

    # статический метод - оторванный от класса и экземпляра
    @staticmethod
    def norm(x, y):
        return x ** 2, y ** 2


v = Vector(1, 2)
# вызов validate
print(v.validate(5))

```

# @staticmethod

Метод, которому "не нужен" класс и его состояние и может использоваться в отрыве от него

```python
class Math:
    @staticmethod
    def sum(a, b):
        return a + b
```

# property

```python
class Person:
    def __init__(self, name, old):
        self.__name = name
        self.__old = old

    def get_old(self):
        return self.__old

    def set_old(self, old):
        self.__old = old

    # завяжет old c сеттером и геттером
    # первый аргумент геттер, второй - сеттер
    # если будет добавлен локальное свойство, будет переопределено
    old = property(get_old, set_old)
    # аналогично
    old = property()
    old = old.getter()
    old = old.setter(set_old)


p = Person('name', 30)
p.old = 35  # установит
print(p.old)  # получит
```

Альтернативный вариант с @

```python
class Person:
    def __init__(self, name, old):
        self.__name = name
        self.__old = old

    # геттеру присваиваем @property
    @property
    def old(self):
        return self.__old

    # здесь декоратор должен иметь имя как геттер + свойство setter
    @old.setter
    def old(self, old):
        self.__old = old

    @old.deleter
    def old(self):
        del self.__old
```

# Декораторы для функций и методов

Для классов декораторов без аргументов, передаваемых в декоратор

```python
class DecoratorClassWithoutArgs:
    def __init__(self, func):
        self.__func = func

    def __call__(self, *args, **kwargs):
        return self.__func(*args, **kwargs)
```

Для классов декораторов с передаваемыми аргументами в декоратор

```python
class DecoratorClassWithArgs:
    # decorator_args - аргументы передаваемые в декоратор
    def __init__(self, decorator_args):
        self.__decorator_args = decorator_args

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
```

# Декоратор класса

```python
def class_decorator(cls):
    class Wrapped(cls):
        def new_method(self):
            return "Новый метод!"

    return Wrapped


@class_decorator
class MyClass:
    def original_method(self):
        return "Оригинальный метод"


obj = MyClass()
print(obj.original_method())  # Оригинальный метод
print(obj.new_method())  # Новый метод!

```

Пример логгера на уровне класса

```python
def log_methods(cls):
    for attr_name, attr_value in cls.__dict__.items():
        if callable(attr_value) and not attr_name.startswith("__"):
            original = attr_value

            def wrapper(self, *args, _original=original, **kwargs):
                print(f"Вызов метода: {attr_name}")
                # внутри цикла нужно аккуратно обращаться с лямбдами/функциями из-за замыканий — 
                # чтобы избежать багов, мы явно передаём original как _original
                return _original(self, *args, **kwargs)

            setattr(cls, attr_name, wrapper)
    return cls


@log_methods
class MyClass:
    def greet(self):
        print("Привет!")


obj = MyClass()
obj.greet()
# Вывод:
# Вызов метода: greet
# Привет!

```

Регистрация классов

```python
registry = {}


def register_class(cls):
    registry[cls.__name__] = cls
    return cls


@register_class
class MyService:
    pass


print(registry)
# {'MyService': <class '__main__.MyService'>}

```

<!--  -->

```python

```