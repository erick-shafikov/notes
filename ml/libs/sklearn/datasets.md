# sklearn.datasets

Встроенные датасеты для быстрого прототипирования и тестирования.

## Табличные датасеты

```python
from sklearn.datasets import (
    load_iris,          # классификация: 150 цветков, 4 признака, 3 класса
    load_digits,        # классификация: 1797 цифр 8×8, 10 классов
    load_wine,          # классификация: 178 вин, 13 признаков, 3 класса
    load_breast_cancer, # классификация: 569 опухолей, 30 признаков, 2 класса
    load_diabetes,      # регрессия: 442 пациента, 10 признаков, целевое — прогрессия болезни
    load_boston,        # регрессия (устарел, removed в 1.2): цены на недвижимость
)

data = load_iris()
X, y = data.data, data.target        # ndarray (150, 4) и (150,)
data.feature_names                   # ['sepal length (cm)', ...]
data.target_names                    # ['setosa', 'versicolor', 'virginica']
data.DESCR                           # текстовое описание датасета

# упрощённая загрузка через return_X_y
X, y = load_iris(return_X_y=True)

# как DataFrame
df = load_iris(as_frame=True).frame  # pandas DataFrame
```

Все `load_*` возвращают `Bunch` — dict-подобный объект с полями `data`, `target`, `feature_names`, `target_names`, `DESCR`.

---

## Изображения

```python
from sklearn.datasets import load_sample_image, load_sample_images

# одно изображение по имени файла
img = load_sample_image("china.jpg")   # ndarray (427, 640, 3), dtype=uint8
img = load_sample_image("flower.jpg")  # ndarray (427, 640, 3), dtype=uint8

# все встроенные изображения сразу
dataset = load_sample_images()
dataset.images        # list of ndarray
dataset.filenames     # ['china.jpg', 'flower.jpg']
```

Для оттенков серого: `gray = img.mean(axis=2)` или `img.mean(axis=2).astype(np.uint8)`.

---

## Генераторы синтетических данных

```python
from sklearn.datasets import make_classification, make_regression, make_blobs

# классификация с настраиваемой структурой
X, y = make_classification(
    n_samples=1000,
    n_features=20,        # всего признаков
    n_informative=5,      # из них информативных
    n_redundant=2,        # линейные комбинации информативных
    n_classes=2,
    random_state=42,
)

# регрессия
X, y = make_regression(
    n_samples=500,
    n_features=10,
    n_informative=5,
    noise=0.1,
    random_state=0,
)

# кластеры (гауссовы облака)
X, y = make_blobs(
    n_samples=300,
    centers=3,            # число кластеров (или массив центров)
    cluster_std=0.5,      # разброс каждого кластера
    random_state=0,
)
```
