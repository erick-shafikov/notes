# Картежи

неизменяемая коллекция данных, нельзя не удалять и менять


```python
# картеж из одного значения
a = (1, )
a = (1, 2)
a = 1, 2
# обращение
a[0]
# распаковка
a, b = (1,2 )
# длина картежей
len(a)
# НЕ создаст копию, а будет ссылаться на тот де b
b = a[:]
# пустой
a = ()
a = tuple()

# объединение картежей
a = ()
a = a + (1, )
# дублирование
b = (0) * 10
# из списка в картеж
a = tuple([1, 2, 3])
a = tuple('string')# (s, t, r, ...)

```
Картеж может быть ключом в словаре


# методы

```python
# найти сколько раз встречается элемент
a.count('element')
# поиск индекса
a.index(1)
# 2 - с какого индекса искать
a.index(1,2)
# 2 - с какого индекса искать 3 - до какого
a.index(1,2,3)
# узнать входит ли элемент
some in a
```

