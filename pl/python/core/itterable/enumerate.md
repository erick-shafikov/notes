Позволяет достать из листов и элемент и индекс


```python
# поменять все двузначные на 0
digs = [1, 2, 3, 4, 5]

for i,d in enumerate(digs):
  if 10 <= abs(d) <= 99:
    digs[i] = 0

prints(digs)
```