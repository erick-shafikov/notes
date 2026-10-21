# matplotlib — основы

```python
import matplotlib.pyplot as plt
import numpy as np
```

---

## plt.plot

Рисует линию по точкам `(x, y)`.

```python
x = np.linspace(0, 2 * np.pi, 100)

plt.plot(x, np.sin(x))              # просто линия
plt.plot(x, np.cos(x), 'r--')       # красный пунктир
plt.plot(x, np.sin(x), label='sin') # с подписью для легенды
plt.legend()
plt.show()
```

Часто используемые параметры:

| параметр | что делает |
|----------|------------|
| `color='red'` / `'r'` | цвет линии |
| `linestyle='--'` / `':'` | стиль линии |
| `linewidth=2` | толщина |
| `marker='o'` | маркер на точках |
| `label='name'` | подпись для `plt.legend()` |

---

## plt.scatter

Точечный график — каждая точка независима (в отличие от `plot`, где точки соединены).

```python
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# c — цвет каждой точки (массив), s — размер, cmap — палитра
plt.scatter(x, y, c=y, s=20, cmap='viridis', alpha=0.7)
plt.colorbar()   # шкала цвета
plt.show()
```

---

## plt.imshow

Отображает матрицу как изображение. Используется для: изображений в оттенках серого, тепловых карт, матриц корреляций.

```python
# оттенки серого (2D массив)
gray = np.random.rand(64, 64)
plt.imshow(gray, cmap='gray')
plt.colorbar()
plt.show()

# RGB изображение (3D массив H×W×3, значения 0–255 или 0.0–1.0)
img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
plt.imshow(img)
plt.axis('off')   # убрать оси
plt.show()
```

Параметры:

| параметр | что делает |
|----------|------------|
| `cmap='gray'` / `'viridis'` | палитра для 2D |
| `vmin`, `vmax` | пределы шкалы цвета |
| `interpolation='nearest'` | без сглаживания пикселей |
| `aspect='auto'` | не сохранять пропорции пикселей |

---

## plt.subplots

Создаёт сетку графиков. Возвращает `fig` (всё окно) и `axes` (массив осей).

```python
# одна строка, 3 столбца
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, ax in enumerate(axes):
    ax.plot(np.random.randn(50))
    ax.set_title(f"График {i}")

plt.tight_layout()   # убирает перекрытие подписей
plt.show()
```

```python
# сетка 2×2
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].imshow(np.random.rand(32, 32), cmap='gray')
axes[0, 1].plot([1, 2, 3])
axes[1, 0].bar(['a', 'b', 'c'], [3, 1, 2])
axes[1, 1].scatter(np.random.randn(50), np.random.randn(50))

plt.tight_layout()
plt.show()
```

`figsize=(width, height)` — размер в дюймах.

---

## plt.hist

Гистограмма распределения.

```python
data = np.random.randn(1000)

plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.show()
```

---

## Подписи и сетка

```python
plt.plot(x, y)

plt.title('Заголовок')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.legend()          # нужен label= в plot/scatter
plt.grid(True)        # сетка
plt.xlim(0, 10)       # пределы по X
plt.ylim(-1, 1)       # пределы по Y
plt.show()
```

---

## Сохранение в файл

```python
fig, ax = plt.subplots()
ax.plot(x, y)

# dpi — разрешение, bbox_inches='tight' — обрезать лишние поля
fig.savefig('plot.png', dpi=150, bbox_inches='tight')
```

---

## plt.colorbar

Добавляет шкалу цвета к последнему `imshow` или `scatter` с `c=`.

```python
im = plt.imshow(matrix, cmap='hot')
plt.colorbar(im, label='Значение')
plt.show()
```
