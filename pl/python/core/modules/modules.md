# встроенные модули 

```python
# переопределение название модуля
import math as mt
# импорт отдельных функций
from math import ceil, pi
# импорт отдельных функций
from math import ceil as math_ceil, pi
# не рекомендуется
from math import *
```

# собственные модули модули 
```python
# folder/mymodule.py - импортируемый модуль
# 
import math
NAME = 'mymodule'

def floor(x):
  print(x)

```

```python
# ex1.py главный файл если на одном уровне
import mymodule

import folder .mymodule
# будут доступны mymodule.floor и mymodule.NAME
# будут доступны mymodule.math.floor вылечить from math import *
```

```python
import sys
sys.path # информация о модуля

if __name__ == '__name__'
# что то сделать
```

# сторонние модули

- pip list - команда для проверки списки библиотек
- pip instal - дял скачивания
- pip instal libs==2 - дял скачивания конкретной версиис м
- pip freeze textfile.txt - распечатает в файл список библиотек

# пакеты

набор файлов

<!--  -->

```python
# файл инициализации
#__init__.py
from courses.python import get_python
from .python import get_python
from . import get_python, get_php, 
from .python import *
# импорт с уровней выше
from ..python import *
```

```python
# контроль внешних модулей
__all__ = ['some_func']

def some_func():
  print()


```


<!--  -->

```python

```
