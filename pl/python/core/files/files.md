# open

функция для открытия файла

```python
# открыть my_file.txt c кодировкой 'utf-8'
file = open('my_file.txt', encoding='utf-8')
# при вызове по 4 символа 
file.read(4)
# читать с 0 символа
file.seek(0)
# вернет позицию
file.tell(0)
# прочитать строку
file.readline()
# все строки (с \n)
file.readlines()
# !!!обязательно закрывать файл
file.close()
```

# exceptions

```python
try:
    file = open('my_file.txt', encoding='utf-8')
    try:
        s = file.readLines()
    finally:
        file.close()
except FileNotFoundError:
    print('нет файла')
except:
    print('другие ошибки')
```

# менеджер контекста

```python

try:
    # откроет и закроет файл
    with file = open('my_file.txt', encoding='utf-8') as file
    s = file.readLines()
    # try:
    #   s = file.readLines()
    # finally:
    #   file.close()
except FileNotFoundError:
    print('нет файла')
except:
    print('другие ошибки')
finally:
    # флаг закрыт или нет
    print(file.close)
```

# запись данных

если открыть несуществующий, то будет создан новый пустой файл

```python
# флаг w - на запись, по умолчанию r
try:
    with file = open('text.txt'.'w') as file:
        # перезаписать данные
        file.write('Hello1\n')
        file.write('Hello2\n')
        file.write('Hello2\n')
except:
    print('ошибка')

```

добавить строки

```python
try:
    # флаг a - на добавление, a+ - и на чтение и на добавления
    with file = open('text.txt'.'a') as file:
        # перезаписать данные
        file.write('Hello1\n')
        file.write('Hello2\n')
        file.write('Hello2\n')
except:
    print('ошибка')
```

запись нескольких строк

```python
try:
    # флаг a - на добавление, a+ - и на чтение и на добавления
    with file = open('text.txt'.'a+') as file:
        # перезаписать данные
        file.writelines('Hello1\n', 'Hello2\n')

except:
    print('ошибка')
```

Бинарный доступ

```python
# специальная библиотека
import pickle

books = [
    ("Евгений Онегин", "Пушкин А.С.", 200),
    ("Муму", "Тургенев И.С.", 250),
    ("Мастер и Маргарита", "Булгаков М.А.", 500),
    ("Мертвые души", "Гоголь Н.В.", 190)
]
# бинарный режим доступа к файлу
file = open('text.txt', 'ab')
# записать
pickle.dump(books, file)
file.close()
# считать
pickle.load(file)
```

<!--  -->

```python

```