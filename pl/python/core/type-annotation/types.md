# type annotation

- аннотация носит рекомендательный характер

# переменные

int, float, str, bool, bytes

```python
cnt: int = 0
```
# функции

```python
def mul2(x: int) -> float:
  return x * 2


print(mul2.__annotation__)
# {x:class float}
```

# typing

## Union

```python
from  typing import Union
# Union[int, float] == int | float
def mul2(x: Union[int, float], y: int | float):
  return x * y
```

## Optional

```python
def show_x(x:float, descr: Optional[str] = None) -> None:
  if descr:
    # ...
  else:
    #...
```

## Any

- любой тип данных

```python
def show_x(x: Any, descr: Optional[str] = None) -> None:
  if descr:
    # ...
  else:
    #...
```
## Final

```python
MAX_VALUE: Final = 10000
```

# list

```python
# все что угодно
list: list = [1, 2, None]
# аннотация списка
list: list[int] = [1, 2 ,3]
# более старый
from typing import List
list: List[int] = [1, 2 ,3]
```

# tuple

```python
addr: tuple[int, str] = (1, 'x')
# для списка из float
elems: tuple[float, ...] 
```

# dict

```python
words: dict[str, int] = {'x': 1, 'y': 2}
```

# set

```python
person: set[str] = {'abc', 'def', 'ghi'}
```

# list, dict, set in func

```python
# лыв варианта Union
def get_positive(digits: list[int | float]) -> list[Union[int, float]]Ж
    # ...
```

# Callable

для аннотации коллбеков

```python
# [int] - аргументы, bool- возвращаемый
def get_digits(flt: Callable[[int], bool], list: list[int]=None) -> list[int]:
  if lst is None:
      return []
  return list(filter(flt, lst))


print(get_digits(lambda x: x % 2 == 0, [1, 2, 3, 4]))
```

<!--  -->

```python

```