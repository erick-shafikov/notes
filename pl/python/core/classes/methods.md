# методы

```python

class Point:
    color = 'red'
    circle = 2

    # обязательный параметр self - ссылка на экземпляр
    # без self будет ошибка
    # метод будет внутри класса
    def set_cords(self, x, y):
        print('something')
        self.x = x
        self.y = y

    def get_cords(self):
        return (self.x, self.y)


a = Point()

# self == a подставит самостоятельно
a.set_cords(1, 2)
```

<!--  -->

```python

```