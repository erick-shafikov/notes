# Set

изменяемый тип данных
множество - непорядочный тип данных

множество уникальных значений. В составе множества может быть только неизменяемые типы данных

```python
# создание
a = {1, 2, 3}
# пустой ({} создаст пустой dict, а не set)
a = set()
# из итерируемого объекта, дубли отбрасываются
a = set([1, 2, 3, 1, 2])  # {1, 2, 3}
# уникальные элементы lst
a = list(set(lst))
# длина
len(a)
# генератор множества
a = {x % 3 for x in range(10)}  # {0, 1, 2}
```

Перебор

с помощью for in и iter

Проверка вхождения — за $O(1)$ в среднем, в отличие от списка ($O(n)$):

```python
3 in a      # True
3 not in a  # False
```

# Методы

```python
b = set()
# добавление
b.add(element)
# добавление нескольких
b.update(some_iter_object)
# удаление без ошибки
b.discard(elem_to_delete)
# удаление, KeyError если элемента нет
b.remove(elem_to_delete)
# удаляет и возвращает произвольный элемент, KeyError если множество пустое
b.pop()
# удаление всех
b.clear()
# поверхностная копия
c = b.copy()
```

# операции над множествами

Операторы (`&`, `|`, `-`, `^`) требуют, чтобы оба операнда были множествами. Методы (`intersection`, `union` и т.д.) принимают любой итерируемый объект и несколько аргументов сразу:

```python
{1, 2} | [2, 3]          # TypeError
{1, 2}.union([2, 3])     # {1, 2, 3}
{1, 2}.union([3], (4,))  # {1, 2, 3, 4}
```

Методы с суффиксом `_update` изменяют множество на месте и возвращают `None` — это аналоги операторов `&=`, `|=`, `-=`, `^=`.

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
# объединение в setA
setA.update(setB)
```

# вычитание

```python
setA = {1, 2, 3, 4}
setB = {3, 4, 5, 6, 7}

setA - setB  # {1, 2}
setB - setA  # {5, 6, 7}
setA -= setB  # {1, 2} в setA
# вернет разность
setA.difference(setB)  # {1, 2}
# разность в setA
setA.difference_update(setB)
```

# симметричная разница

взять только уникальные значения из двух, без общих

```python
setA = {1, 2, 3, 4}
setB = {3, 4, 5, 6, 7}

setA ^ setB  # {1,2,5,6,7}
setA ^= setB  # {1,2,5,6,7} в setA
# вернет симметричную разницу
setA.symmetric_difference(setB)
# симметричная разница в setA
setA.symmetric_difference_update(setB)
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

setB < setA   # True, строгое подмножество
setB <= setA  # True, подмножество (может совпадать)
setA > setB   # True, строгое надмножество
setA >= setB  # True, надмножество
# ну если одно множество не является подмножеством вернет False
```

# issubset, issuperset, isdisjoint

`issubset` — аналог `<=`, `issuperset` — аналог `>=`, но принимают любой итерируемый объект.

```python
s1 = "abc"
s2 = "aabbccdd"

if set(s1).issubset(s2):
    print("Все символы s1 есть в s2")
else:
    print("Не все символы s1 есть в s2")

set(s2).issuperset(s1)  # True

# нет ни одного общего элемента
{1, 2}.isdisjoint({3, 4})  # True
{1, 2}.isdisjoint({2, 3})  # False
```

# frozenset

Неизменяемое множество: есть все методы, которые не меняют множество (`union`, `intersection`, `issubset`, `copy` …), но нет `add`, `remove`, `update` и других `_update`. Так как оно хешируемое, его можно класть в другое множество или использовать как ключ словаря.

```python
fs = frozenset([1, 2, 3])
fs.add(4)  # AttributeError
{fs, frozenset([4])}  # множество множеств
d = {frozenset({"a", "b"}): 1}
```
