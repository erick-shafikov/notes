```python
import torch

t = torch.FloatTensor([1, 2, 3, 10, 20, 30])
a = torch.tensor([])
#
t.sum()  # сумма
t.sum(dim=0)  # сумма по столбцам
t.sum(dim=1)  # сумма по строкам
t.prod(dim=1)  # произведение по строкам
t.mean()  # среднее
t.max()
# для многомерных
t.amax(dim=1)  # вернет строку
t.mean()
torch.abs(a)  # модуль вернет, не изменит
t.abs_()  # изменит текущий, работают inplace методы
torch.log(a)  # логарифм
# тригонометрия
t.sin()
t.sin_()
torch.sin(a)
# статистические
torch.median(a)
t.median()
torch.var(a)
t.var()
torch.corrcoef(a)  # коэффициент корреляции Пирсона
t.corrcoef()
t.cov()
# преобразование к числу
# так как torch возвращает не число, а тензор, что бы преобрзовать к числу:
t.sin().item()
```