- полиморфизм - возможность работать с разными классами с единым интерфейсом

```python
# абстрактный класс
class Geom:
    def get_pr(self):
        raise NotImplementedError('not implemented')


class Rectangle(Geom):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    # в каждом классе нужно определить get_pr
    def get_pr(self):
        return 2 * (self.w + self.h)


class Square(Geom):
    def __init__(self, a):
        self.a = a

    def get_pr(self):
        return 4 * self.a


class Triangle(Geom):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def get_pr(self):
        return self.a + self.b + self.c


r1 = Rectangle(1, 2)
r2 = Rectangle(3, 4)
s1 = Square(10)
s2 = Square(20)

geom = [r1, r2, s1, s2]

for g in geom:
    print(g.get_pr())
```

<!--  -->

```python

```