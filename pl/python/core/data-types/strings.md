Строки

Неизменяемый тип данных

```python

# \n - переход но новую строку
text = '''Много 
Строк'''


```

# Операции

```python
# сложение строк
string1 = 'a'
string2 = 'b'
x = string1 + string2  # 'ab'

# Функция превращает строку аргумент
a = 5
str(a)

# повтор 
y = string1 * 2

# длин строки
len(str)

# вхождение
if 'substr' in string1:
    pass

# сравнение строк
print(string1 == string2)

# код буквы в ASCII
ord('x')

# взять букву
string = 'some string'
s1 = string[1]
s2 = 'some string'[1]

# обращение к последнему
s3 = string[len(str) - 1]
s4 = string[-1]

# срез строки от второго до 3 не включая
sub = string[1:3]
sub1 = string[4:]  # от четвертого
sub2 = string[:4]  # до четвертого
sub3 = string[2:-2]  # от -2 до 2
sub4 = string[2:10:2]  # c шагом два
sub5 = string[1::2]
sub6 = string[::2]  # все через 1
sub7 = string[::-2]  # все через 1 с конца
# равенство
print(string == string[:])
# переворот строки
print(string[::-1])
```

# Методы строк

```python
str.upper()  # все заглавные, строка сама не меняется
str.lower()  # все прописные
str.count('substring', startIndex,
          finishIndex)  # вернуть количество повторений startIndex - начиная с какого, finishIndex - до какого 
str.find('substring', startIndex, finishIndex)  # на каком индексе
str.rfind('substring', startIndex, finishIndex)  # сзади
str.index('substring', startIndex, finishIndex)  # прокинет ошибку при не нахождении
str.replace('replace', 'new value', number_of_replacement)  # менять символы
str.isalpha()  # является ли строкой состоящей только из букв
str.isdigit()  # является ли числом
str.rjust(number_to_fill, symbol_to_fill)  # дополнит строку исходно длины до строки длиной number_to_fill 
str.ljust(number_to_fill, symbol_to_fill)  # дополнит строку исходно длины до строки длиной number_to_fill слева
str.split(separator,
          maxsplit=-1)  # вернет коллекцию подстрок maxsplit - максимальное количество элементов в полученом списке
str.join([])  # вернет строку из коллекции
", ".join([1, 2, 3, 4])  # в начале идет разделитель
str.strip()  # удаляет все символы (пробелы и )
str.rstrip()
str.lstrip()
```

# экранированные символы

символ перевода строки влияет на длину

```python


'\t'  # знак табуляции
'\\'  # знак \
'\''  # апостроф
'\"'  # двойная кавычка
'\a'  # звуковой сигнал
'\b'  # эмуляция клавиши backspace
'\f'  # перевод формата
'\r'  # возврат каретки
'\t'  # горизонтальная табуляция
'\v'  # вертикальная табуляция
'\0'  # символ Null
'\xhh'  # символ 16-кодом
'\ooo'  # символ 8-кодом
'\N{id}'  # идентификатор таблицы Unicode
'\Uhhhh'  # 16 битный идентификатор таблицы Unicode
'\Uhhhhhhhh'  # 32 битный идентификатор таблицы Unicode
# знак кавычек
s = "something \"second\""
# сырая строка
r'здесь не надо ничего экранировать'
```

# метод Format и f-строки

```python

age = 18
name = 'Сергей'
msg = 'Меня зовут {0} мне {1} лет'.format(name, age)  # Меня зовут Сергей мне 18 лет

# или с именованными аргументами
'Меня зовут {fio} мне {old} лет'.format(fio=name, old=age)

```

f-строки

```python
age = 18
name = 'Сергей'
f'Меня зовут {name} мне {age} лет'
# внутри можно вызывать любую функцию

# добавить вначале нули
f'{1:02}'  # 01
f'{1:03}'  # 001
```