# Set

изменяемый тип данных
множество - непорядочный тип данных

множество уникальных значений. В составе множества может быть только неизменяемые типы данных

```python
# создание
a = {1, 2, 3}
# пустой
a = set()
# пустой
a = set([1, 2, 3, 1, 2])  # 1,2,3
# уникальные элементы lst
a = list(set(lst))
# длина
len(a)
```

Перебор

с помощью for in и iter

# Методы

```python
b = set()
# добавление
b.add(element)
# добавление нескольких
b.update(some_iter_object)
# удаление без ошибки
b.discard(elem_to_delete)
# удаление с ошибкой
b.remove()
# удаление с ошибкой
b.pop()
# удаление всех
b.clear()
```

# операции над множествами

## пересечение

```python
setA = {1, 2, 3, 4}
setB = {3, 4, 5, 6, 7}
# 
setA & setB  # {3, 4}
setA &= setB
# вернет пересечение
setA.intersection(setB)
# вернет пересечение в setA
setA.intersection_update(setB)
```

# объединение

```python
setA = {1, 2, 3, 4}
setB = {3, 4, 5, 6, 7}
# 
setA | setB  # {1,2,3,4,5,6,7}
setA |= setB  # {1,2,3,4,5,6,7} в setA
# вернет объединение
setA.union(setB)
```

# вычитание

```python
setA = {1, 2, 3, 4}
setB = {3, 4, 5, 6, 7}

setA - setB  # {1, 2}
setB - setA  # {3, 4}
setA -= setB  # {1, 2}
setB -= setA  # {3, 4}
```

# симметричная разница

взять только уникальные значения из двух, без общих

```python
setA = {1, 2, 3, 4}
setB = {3, 4, 5, 6, 7}

setA ^ setB  # {1,2,5,6,7}
```

# сравнения

```python
# равенство
setA = {7, 6, 5, 4, 3}
setB = {3, 4, 5, 6, 7}
setA == setB  # True

# сравнение
setA = {7, 6, 5, 4, 3}
setB = {3, 4, 5}

setB < setA  # True
# ну если одно множество не является подмножеством вернет False
```

# issubset

```python
s1 = "abc"
s2 = "aabbccdd"

if set(s1).issubset(s2):
    print("Все символы s1 есть в s2")
else:
    print("Не все символы s1 есть в s2")

```

<!--  -->

```python

```