Советы по улучшению работы с изображениями:

- использование правильных форматов
- компрессия изображений
- lazy loading
- адаптивные изображения
- использование cdn

Использования изображения как фон:

# background

это короткая запись для background-clip + background-color + background-image + background-origin + background-position + background-repeat + background-size + background-attachment

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
.background-repeat {
  background-repeat: repeat; // по умолчанию повтор включен
  background-repeat: repeat-x; // repeat no-repeat
  background-repeat: repeat-y; // no-repeat repeat
  background-repeat: space; // будет заполнено не обрезая изображения
  background-repeat: round; // по рамке
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
.background-size {
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

## -------------------------------------------------------

## background-blend-mode

Определяет как будут смешиваться наслаиваемые цвета и изображения

как background-image будет смешиваться с вышележащими слоями

правило смешивания наслаивающих изображений и фонов https://developer.mozilla.org/en-US/docs/Web/CSS/blend-mode

Значений всего 16

```scss
 {
  background-blend-mode: darken, luminosity;
}
```

# Маски clip-path:

## clip-path

```scss
.clip-path {
  clip-path: none;

  /* Значения <clip-source> */
  clip-path: url(resources.svg#c1);

  /* Значения <geometry-box> */
  clip-path: margin-box;
  clip-path: border-box;
  clip-path: padding-box;
  clip-path: content-box;
  clip-path: fill-box;
  clip-path: stroke-box;
  clip-path: view-box;

  /* Значения <basic-shape> */
  clip-path: inset(100px 50px); //Определяет внутренний прямоугольник.
  clip-path: circle(
    50px at 0 100px
  ); //Определяет окружность, используя радиус и расположение.
  clip-path: ellipse(
    50px 60px at 0 10% 20%
  ); //Определяет эллипс, используя два радиуса и расположение
  clip-path: polygon(
    50% 0%,
    100% 50%,
    50% 100%,
    0% 50%
  ); // Определяет многоугольник, используя стиль заполнения фигуры и набор вершин.
  clip-path: path(
    "M0.5,1 C0.5,1,0,0.7,0,0.3 A0.25,0.25,1,1,1,0.5,0.3 A0.25,0.25,1,1,1,1,0.3 C1,0.7,0.5,1,0.5,1 Z"
  ); //Определяет фигуру, используя объявление SVG фигуры и правило заполнения

  /* Комбинация значений границ и формы блока */
  clip-path: padding-box circle(50px at 0 100px);
}
```

```scss
.polygon {
  clip-path: polygon(
    0 0,
    100% 0,
    100% 50%,
    0 100%
  ); //- обрезает картинку по координатам, относительно изображение
}
```

## clip-rule

nonzero | evenodd настрой выбора пикселей для вычета

# mask

краткая запись следующих свойств нужная для маскирования изображения:

mask = mask-clip + mask-composite + mask-image + mask-mode + mask-origin + mask-position + mask-repeat + mask-size

### -webkit-mask-composite

альтернатива mask-composite, так же как и -webkit-mask-position-x, -webkit-mask-position-yNon-standard, -webkit-mask-repeat-xNon-standard, -webkit-mask-repeat-yNon-standard

## mask-clip

mask-clip определяет область применения маски

определяет область применения маски

```scss
 {
  mask-clip: content-box;
  mask-clip: padding-box;
  mask-clip: border-box;
  mask-clip: fill-box;
  mask-clip: stroke-box;
  mask-clip: view-box;

  /* Keyword values */
  mask-clip: no-clip;

  /* Non-standard keyword values */
  -webkit-mask-clip: border;
  -webkit-mask-clip: padding;
  -webkit-mask-clip: content;
  -webkit-mask-clip: text;

  /* Multiple values */
  mask-clip: padding-box, no-clip;
  mask-clip: view-box, fill-box, border-box;
}
```

## mask-image

ресурс для маски

```scss
 {
  mask-image: url(masks.svg#mask1);

  /* <image> values */
  mask-image: linear-gradient(rgb(0 0 0 / 100%), transparent);
  mask-image: image(url(mask.png), skyblue);

  /* Multiple values */
  mask-image: image(url(mask.png), skyblue), linear-gradient(rgb(0 0 0 / 100%), transparent);
}
```

## mask-mode

alpha | luminance | match-source

## mask-origin

определяет расположение начала

```scss
 {
  mask-origin: content-box; // Положение указывается относительно границы поля.
  mask-origin: padding-box; //Положение указывается относительно ограничивающей рамки объекта.
  mask-origin: border-box; //Положение указывается относительно ограничивающей рамки штриха.
  mask-origin: fill-box; //Использует ближайший вьюпорт SVG
  mask-origin: stroke-box; //
  mask-origin: view-box;

  /* Multiple values */
  mask-origin: padding-box, content-box;
  mask-origin: view-box, fill-box, border-box;

  /* Non-standard keyword values */
  -webkit-mask-origin: content; //content-box
  -webkit-mask-origin: padding; //padding-box.
  -webkit-mask-origin: border; //border-box.
}
```

## mask-position

25% 75% позиция top/left

## mask-repeat

Определение повторение маски

```scss
 {
  mask-repeat: repeat-x;
  mask-repeat: repeat-y;
  mask-repeat: repeat;
  mask-repeat: space;
  mask-repeat: round;
  mask-repeat: no-repeat;

  /* Two-value syntax: horizontal | vertical */
  mask-repeat: repeat space;
  mask-repeat: repeat repeat;
  mask-repeat: round space;
  mask-repeat: no-repeat round;
}
```

## mask-size

Размер маски

```scss
 {
  /* Keywords syntax */
  mask-size: cover;
  mask-size: contain;

  /* One-value syntax */
  /* the width of the image (height set to 'auto') */
  mask-size: 50%;
  mask-size: 3em;
  mask-size: 12px;
  mask-size: auto;

  /* Two-value syntax */
  /* first value: width of the image, second value: height */
  mask-size: 50% auto;
  mask-size: 3em 25%;
  mask-size: auto 6px;
  mask-size: auto auto;

  /* Multiple values */
  /* Do not confuse this with mask-size: auto auto */
  mask-size: auto, auto;
  mask-size: 50%, 25%, 25%;
  mask-size: 6px, auto, contain;
}
```

## mask-type

luminance | alpha тип маски

# Фильтры

## filter

Добавляет фильтры на изображения

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

# object-fit

Позволяет тегу img определить размеры относительно контейнера

```scss
.object-fit {
  object-fit: fill; //заполняет весь контейнер, меняя свои пропорции
  object-fit: contain; //растянет под контейнер, но оставит пропорции
  object-fit: cover; //оставит пропорции, но поместит в контейнер часть изображения
  object-fit: none; //подстроится под изображение
  object-fit: scale-down; //выберет меньший между none и contain
}
```

# object-position

расположит изображение в контейнере

```scss
 {
  object-position: center top;
  object-position: 100px 50px;
}
```

- [тени](./block-model.md#box-shamask-border-modedow)

# image-свойства

- image-orientation: none | from-image позволяет клиенту автоматически перевернуть изображение
- image-rendering: auto | crisp-edges | pixelated позволяет сгладить края при возникновении пикселей в изображении
- image-resolution (экспериментальное) управление качеством

# -moz-force-broken-image-icon (-)

отображать или нет у изображений, которые не удалось загрузить иконку картинки, у которых есть alt атрибут

```scss
.moz-force-broken-image-icon {
  moz-force-broken-image-icon: 0; //нет
  moz-force-broken-image-icon: 1; //да
}
```

# Градиенты

Градиенты могут быть использованы, там где используются изображения.

## linear-gradient()

linear-gradient() - создает линейный градиент

```scss
.simple-linear {
  //двухцветные с плавным переходом
  background: linear-gradient(blue, pink);
  background: linear-gradient(#e66465, #9198e5);
  // по умолчанию будут равно распределены
  background: linear-gradient(red, yellow, blue, orange);
  //распределение в неравных пропорциях
  background: linear-gradient(to left, lime 28px, red 77%, cyan);
  //конкретные промежутки
  background: linear-gradient(
    to left,
    lime 25%,
    red 25% 50%,
    cyan 50% 75%,
    yellow 75%
  );
  background: linear-gradient(to left, #333, #333 50%, #eee 75%, #333 75%);
  // для резкого перехода проценты в сумме должны быть равны 100
  background: linear-gradient(to bottom left, cyan 50%, palegoldenrod 50%);
  // подсказка для перехода 10% займет blue
  background: linear-gradient(blue, 10%, pink);
}

// поменять направление
.horizontal-gradient {
  background: linear-gradient(to right, blue, pink);
  // диагональный
  background: linear-gradient(to bottom right, blue, pink);
  // использование углов
  background: linear-gradient(70deg, blue, pink);
  background: linear-gradient(0.25turn, #3f87a6, #ebf8e1, #f69d3c);
}
```

```scss
// можно комбинировать с изображениями
.layered-image {
  background: linear-gradient(to right, transparent, mistyrose),
    url("critters.png");
}
```

Наслоение нескольких друг на друга

```scss
 {
  background: linear-gradient(
      217deg,
      rgba(255, 0, 0, 0.8),
      rgba(255, 0, 0, 0) 70.71%
    ), linear-gradient(127deg, rgba(0, 255, 0, 0.8), rgba(0, 255, 0, 0) 70.71%),
    linear-gradient(336deg, rgba(0, 0, 255, 0.8), rgba(0, 0, 255, 0) 70.71%);
}
```

## radial-gradient()

```scss
// расположение центра
.radial-gradient {
  background: radial-gradient(at 0% 30%, red 10px, yellow 30%, #1e90ff 50%);
}
//размер определяется расстоянием от начальной точки (центра) до ближайшей стороны блока.
.radial-ellipse-side {
  background: radial-gradient(
    ellipse closest-side,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
  // устанавливает размер градиента значением расстояния от начальной точки до самого дальнего угла блока.
  background: radial-gradient(
    ellipse farthest-corner at 90% 90%,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
  // по x - 25% от контейнера, по y -75%
  background: radial-gradient(
    circle closest-side at 25% 75%,
    red,
    yellow 10%,
    #1e90ff 50%,
    beige
  );
}
```

## conic-gradient()

conic-gradient() - создает круговой градиент

```scss
.conic-gradient {
  // смещение центра
  background: conic-gradient(at 0% 30%, red 10%, yellow 30%, #1e90ff 50%);
  // поворот поворота
  background: conic-gradient(
    from 45deg,
    red,
    orange,
    yellow,
    green,
    blue,
    purple
  );
}
```

## repeating-linear-gradient

repeating-linear-gradient() - линии

```scss
.repeating-linear {
  background: repeating-linear-gradient(
    -45deg,
    red,
    red 5px,
    blue 5px,
    blue 10px
  );
}
```

## repeating-conic-gradient()

лучи из центра

## repeating-radial-gradient() - круги из центра

- при создании должно быть указано как минимум два цвета

# BP

## BP. Дефолтный стиль для img

!!! При работе с изображениями

```scss
//запретить вытекание за родительский контейнер
 {
  max-width: 100%;
  height: auto;
}
```

## BP. Центрирование изображения

```css
img {
  display: block;
  margin: 0 auto;
}
```
