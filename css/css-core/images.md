Советы по улучшению работы с изображениями:

- использование правильных форматов
- компрессия изображений
- lazy loading
- адаптивные изображения
- использование cdn

# Графика и изображения

Использования изображения как фон:

# background

это короткая запись для background-attachment + background-clip + background-color + background-image + background-origin + background-position + background-repeat + background-size

```scss
 {
  //два цвета смешаются ровно посередине
  background: linear-gradient(#e66465, #9198e5);
  // множественный градиент
  background: linear-gradient(
      217deg,
      rgba(255, 0, 0, 0.8),
      rgba(255, 0, 0, 0) 70.71%
    ), linear-gradient(127deg, rgba(0, 255, 0, 0.8), rgba(0, 255, 0, 0) 70.71%),
    linear-gradient(336deg, rgba(0, 0, 255, 0.8), rgba(0, 0, 255, 0) 70.71%);
}
```

Сокращенная запись

```scss
.box {
  background: linear-gradient(
        105deg,
        rgb(255 255 255 / 20%) 39%,
        rgb(51 56 57 / 100%) 96%
      ) center center / 400px 200px no-repeat, url(big-star.png) center
      no-repeat, rebeccapurple;
}
```

## background-attachment

Определяет поведения заднего фона при прокрутке

```scss
 {
  background-attachment: scroll; //изображение позади будет прокручиваться
  background-attachment: fixed; //изображение позади не будет прокручиваться
  background-attachment: local; //в зависимости от прокручивания контента позади которого будет изображение
}
```

## background-clip

Настраивает как будет обрезаться изображение, которое находится позади

```scss
 {
  background-clip: border-box; //до края границы
  background-clip: padding-box; // до края отступа
  background-clip: content-box; // внутри содержимого
  background-clip: text; //обрезка текстом
}
```

## background-image

Может быть как градиентом так и изображением

```scss
 {
  background-image: linear-gradient(black, white);
  background-image: url("image.png");

  background-image: url(image1.png), url(image2.png), url(image3.png),
    url(image1.png); // несколько изображений

  //Создание изображения с наложением градиента
  background-image: linear-gradient(
      rgba($color-secondary, 0.93),
      rgba($color-secondary, 0.93)
    ), url("../img/hero.jpeg");
  //пример с распределением градиента
  background-image: linear-gradient(
      105deg,
      rgba($color-white, 0.9) 0%,
      rgba($color-white, 0.9) 50%,
      rgba($color-white, 0.9),
      transparent 50%
    ), url("../img/nat-10.jpg");
}
```

## background-origin

как расположить изображение относительно рамок и контента

```scss
 {
  background-repeat: no-repeat; // сначала нужно отключить повтор изображения
  //позиционирование
  background-origin: border-box; //растянуть по всему контейнеру, Фон располагается относительно рамки.
  background-origin: padding-box; //не включая рамки, Фон расположен относительно поля отступа.
  background-origin: content-box; //только по границам контента, Фон располагается относительно поля содержимого.
}
```

## background-position

```scss
 {
  // прилепить к краям
  background-position: top;
  background-position: bottom;
  background-position: left;
  background-position: right;
  background-position: center;
  // сдвиг в процентах и единицах
  background-position: 25% 75%;
  background-position: 0 0;
  background-position: 1cm 2cm;
  background-position: 10ch 8em;
  // точное расположение относительно краев
  background-position: bottom 10px right 20px;
  background-position: right 3em bottom 10px;
  background-position: bottom 10px right;
  background-position: top right 10px;
  // для нескольких изображений
  background-position: 0 0, center;

  // несколько изображений
  background-image: url(image1.png), url(image2.png), url(image3.png),
    url(image1.png);
  background-repeat: no-repeat, repeat-x, repeat; // для image1.png будет применено no-repeat так как свойства применяются циклично
}
```

### background-position-x и background-position-y

определяет горизонтальную позицию изображения

```scss
 {
  background-position-x: left;
  background-position-x: center;
  background-position-x: right;

  /* <percentage> values */
  background-position-x: 25%;

  /* <length> values */
  background-position-x: 0px;
  background-position-x: 1cm;
  background-position-x: 8em;

  /* Side-relative values */
  background-position-x: right 3px;
  background-position-x: left 25%;

  /* Multiple values */
  background-position-x: 0px, center;
}
```

## background-repeat

```scss
 {
  background-repeat: repeat; // по умолчанию повтор включен
  background-repeat: repeat-x; // repeat no-repeat
  background-repeat: repeat-y; // no-repeat repeat
  background-repeat: space; // будет заполнено не обрезая изображения
  background-repeat: round; //
  background-repeat: no-repeat; // отключить повтор

  // варианты повтора изображения можно задавать для вертикальной и горизонтальной осей
  background-repeat: repeat space;
  background-repeat: repeat repeat;
  background-repeat: round space;
  background-repeat: no-repeat round;
}
```

## background-size

Управление размером изображения

```scss
 {
  background-size: cover; // cover - растянет изображение по всему блоку сохраняя пропорции, но обрежет при надобности
  background-size: contain; //  contain - растянет по всем блоку но изменит пропорции

  /* Указано одно значение - ширина изображения, */
  /* высота в таком случае устанавливается в auto */
  background-size: 50%;
  background-size: 3em;
  background-size: 12px;
  background-size: auto; // растягивает сохраняя пропорции

  // два значения - по горизонтали и вертикали
  background-size: 50% auto;
  background-size: 3em 25%;
  background-size: auto 6px;
  background-size: auto auto;

  /* Значения для нескольких фонов */
  /* Не путайте такую запись с background-size: auto auto */
  background-size: auto, auto;
  background-size: 50%, 25%, 25%;
  background-size: 6px, auto, contain;
}
```

# background-blend-mode

Определяет как будут смешиваться наслаиваемые цвета и изображения

```scss
 {
  background-blend-mode: darken | luminosity...;
}
```

# Маски clip-path:

- [использование свойства clip-path](./css-props#clip-path)
- - [clip-rule: nonzero | evenodd настрой выбора пикселей для вычета]

mask - краткая запись следующих свойств нужная для маскирования изображения:

- - [mask-clip определяет область применения маски](./css-props.md#mask-clip)
- - [mask-image определяет url маски](./css-props.md/#mask-image)
- - [mask-mode: alpha | luminance | match-source]
- - [mask-origin определяет расположение начала](./css-props.md/#mask-origin)
- - [mask-position: 25% 75% позиция top/left ]
- - [mask-repeat степень повторения](./css-props.md/#mask-repeat)
- - [mask-size размер](./css-props.md/#mask-size)
- - [mask-type: luminance | alpha тип маски ]

mask-border (экспериментальное) краткая запись следующих свойств позволяет создать маску для границ:

- - mask-border-mode: luminance | alpha использование яркости или альфа-значения в качестве маски
- - mask-border-outset: 7px 12px 14px 5px; отступы
- - mask-border-repeat: stretch | repeat | round | space применение
- - mask-border-slice: 7 12 14 5
- - mask-border-source: url(image.jpg); источник
- - mask-border-width: 5% 2em 10% auto; размеры

# Фильтры

- [использование свойства filter](./css-props#filter)
- [функции для свойства filter](./functions.md/#filter-функции)
- [использование фильтра, который будет применяться к контенту, который находится поверх background-color или image](./css-props.md/#backdrop-filter)
- [для упаковки тега img в контейнер](./css-props.md#object-fit)
- [расположение изображения в контейнере object-position](./css-props.md/#object-position)
- [тени](./css-props.md#box-shadow)
- Текст, который залит фоном:
- - [обрезка фона под текст](./css-props.md#webkit-background-clip)
- - [заливка](./css-props.md#webkit-text-fill-color)

# image-свойства

- image-orientation: none | from-image позволяет клиенту автоматически перевернуть изображение
- image-rendering: auto | crisp-edges | pixelated позволяет сгладить края при возникновении пикселей в изображении
- image-resolution (экспериментальное) управление качеством

# Градиенты

Градиенты могут быть использованы, там где используются изображения.

[gradient() - что бы создать функцию ](./functions.md#градиенты)

- при создании должно быть указано как минимум два цвета

## BP

### BP. Дефолтный стиль для img

!!! При работе с изображениями

```scss
//запретить вытекание за родительский контейнер
 {
  max-width: 100%;
  height: auto;
}
```

### BP. Центрирование изображения

```css
img {
  display: block;
  margin: 0 auto;
}
```
