# Менеджер контекста

```python
# пример без менеджера контекста
fp = None

try:
    fp = open('mygile.txt')
    for t in fp:
        print(t)
except Exception as e:
    print(e)
finally:
    # обязательно закрыть
    if fp is not None:
        fp.close()
```

# with

- __enter()__ - срабатывает при создании объекта менеджера контекста
- __exit()__ - срабатывает в момент закрытия

```python
fp = None

try:
    # __enter()__ вызывается здесь и возвращает fp
    with open('mygile.txt') as fp:
        for t in fp:
            print(t)
except Exception as e:
    # __exit()__ 
    print(e)
```

```python
class DefendedDVector:
    def __init__(self, v):
        # сохраняем в переменную
        self.__v = v

    def __enter__(self):
        # создаем резервную копию для результата работы операции
        self.__temp = self.__v[:]
        return self.__temp

    def __exit__(self, exc_type, exc_val, exc_tb):
        # если все ок
        if exc_type is None:
            self.__v[:] = self.__temp

        # не изменять через __temp
        return False


# смысл при сумме двух массивов пробрасывать ошибку
v1 = [1, 2, 3]
v2 = [2, 3]

try:
    with DefendedDVector(v1) as dv:
        for i, a in enumerate(dv):
            dv[i] += v2[i]
except:
    print('error')

# без менеджера контекста v1 изменился
print(v1)
```