hash - функция дял хеширование объектов

- можно вычислять только для неизменяемого объекта
- используется в ключах словаря

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # для этого класса hash работать не будет
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


```



