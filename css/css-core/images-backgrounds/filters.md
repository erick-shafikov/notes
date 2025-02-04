# Фильтры и размытие

## filter - функции:

Добавляет фильтры на изображения, применять к изображению

```scss
{
  filter: url(resources.svg);
  filter: blur(5px);
  filter: brightness(0.4);
  filter: contrast(200%);
  filter: drop-shadow(16px 16px 20px blue);
  filter: grayscale(50%);
  filter: hue-rotate(90deg);
  filter: invert(75%);
  filter: opacity(25%);
  filter: saturate(30%);
  filter: sepia(60%);
  // fill – заливка
  fill: currentColor; заливка цветом
}
```

```scss
img {
}

.blur {
  filter: blur(10px);
}
```

```html
<div class="box"><img src="balloons.jpg" alt="balloons" class="blur" /></div>
```

Фильтр можно добавить к объектам, то есть к самой тени

```scss
p {
  border: 5px dashed red;
}

.filter {
  filter: drop-shadow(5px 5px 1px rgb(0 0 0 / 70%));
}

.box-shadow {
  box-shadow: 5px 5px 1px rgb(0 0 0 / 70%);
}
```

- [функции для свойства filter](./functions/filters-func.md)

### blur()

функция размытия изображения

```scss
.blur {
  filter: blur(0); /* Без эффекта */
  filter: blur(8px); /* Размытие с радиусом 8px */
  filter: blur(1.17rem); /* Размытие с радиусом 1.17rem */
}
```

## backdrop-filter

позволяет применить фильтр к контенту, который находится поверх контейнера с background-color или background-image
использование фильтра, который будет применяться к контенту, который находится поверх background-color или image

```scss
 {
  backdrop-filter: none;

  /* фильтр URL в SVG */
  backdrop-filter: url(commonfilters.svg#filter);

  /* значения <filter-function> */
  backdrop-filter: blur(2px);
  backdrop-filter: brightness(60%);
  backdrop-filter: contrast(40%);
  backdrop-filter: drop-shadow(4px 4px 10px blue);
  backdrop-filter: grayscale(30%);
  backdrop-filter: hue-rotate(120deg);
  backdrop-filter: invert(70%);
  backdrop-filter: opacity(20%);
  backdrop-filter: sepia(90%);
  backdrop-filter: saturate(80%);

  /* Несколько фильтров */
  backdrop-filter: url(filters.svg#filter) blur(4px) saturate(150%);
}
```

Пример контента с изображением, фон которого будет размыт

```scss
// контент фон которого будет размыт
.box {
  background-color: rgba(255, 255, 255, 0.3);
  -webkit-backdrop-filter: blur(10px);
  backdrop-filter: blur(10px);
}

// изображение
img {
  background-image: url("anemones.jpg");
  background-position: center center;
  background-repeat: no-repeat;
  background-size: cover;
}

.container {
  align-items: center;
  display: flex;
  justify-content: center;
  height: 100%;
  width: 100%;
}
```
