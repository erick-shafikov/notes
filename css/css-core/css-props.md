<!-- backdrop-filter ----------------------------------------------------------------------------------------------------------------------->

# backdrop-filter

позволяет применить фильтр к контенту, который находится поверх контейнера с background-color или background-image

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

<!-- break-after (break-before, break-inside)  --------------------------------------------------------------------------------------------------------------------------->

# break-after (break-before, break-inside)

Применяется для определения разрыва страницы при печати а также для сетки из колонок

break-inside - управление разрывами внутри колонок
break-before, break-inside - до и после

```scss
 {
  break-after: auto; //не будет форсировать разрыв
  break-after: avoid; //избегать любых переносов до/после блока с
  break-after: always;
  break-after: all;

  /* Page break values */
  break-after: avoid-page;
  break-after: page;
  break-after: left;
  break-after: right;
  break-after: recto;
  break-after: verso;

  /* Column break values */
  break-after: avoid-column;
  break-after: column;

  /* Region break values */
  break-after: avoid-region;
  break-after: region;
}
```

<!-- clip-path ------------------------------------------------------------------------------------------------------------------->

# clip-path

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

<!-- contain-intrinsic- ---------------------------------------------------------------------------------------------------------------------------->

# contain-intrinsic-block-size | block-height | inline-size | intrinsic-size | intrinsic-width

Настройка размеров блочных и строчных элементов при ограничении

contain-intrinsic-size = contain-intrinsic-width + contain-intrinsic-height

```scss
.contain-intrinsic {
  contain-intrinsic-block-size: 1000px;
  contain-intrinsic-block-size: 10rem;
  contain-intrinsic-height: 1000px;
  contain-intrinsic-height: 10rem;
  contain-intrinsic-inline-size: 1000px;
  contain-intrinsic-inline-size: 10rem;

  /* auto <length> */
  contain-intrinsic-block-size: auto 300px;
  contain-intrinsic-height: auto 300px;
  contain-intrinsic-inline-size: auto 300px;
}
```

<!-- content ------------------------------------------------------------------------------------------------------------------------------->

# content

заменяет элемент сгенерированным значением

```scss
.elem:after {
  content: normal;
  content: none;

  /* Значение <url>  */
  content: url("http://www.example.com/test.png");

  /* Значение <image>  */
  content: linear-gradient(#e66465, #9198e5);

  /* Указанные ниже значения могут быть применены только к сгенерированному контенту с использованием ::before и ::after */

  /* Значение <string>  */
  content: "prefix";

  /* Значения <counter> */
  content: counter(chapter_counter);
  content: counters(section_counter, ".");

  /* Значение attr() связано со значением атрибута HTML */
  content: attr(value string);

  /* Значения <quote> */
  content: open-quote;
  content: close-quote;
  content: no-open-quote;
  content: no-close-quote;

  /* Несколько значений могут использоваться вместе */
  content: open-quote chapter_counter;
}
```

Пример с возможность заменить

```scss
#replaced {
  content: url("mdn.svg");
}

#replaced::after {
  /* не будет отображаться, если замена элемента поддерживается */
  content: " (" attr(id) ")";
}
```

<!-- filter ------------------------------------------------------------------------------------------------------------------->

# filter

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

<!-- mask ---------------------------------------------------------------------------------------------------------------------------------->

# mask

mask = mask-clip + mask-composite + mask-image + mask-mode + mask-origin + mask-position + mask-repeat + mask-size

<!-- mask-clip ----------------------------------------------------------------------------------------------------------------------------->

# mask-clip

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

<!-- mask-image ---------------------------------------------------------------------------------------------------------------------------->

# mask-image

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

<!-- mask-origin --------------------------------------------------------------------------------------------------------------------------->

# mask-origin

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

<!-- mask-repeat ---------------------------------------------------------------------------------------------------------------------------->

# mask-repeat

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

<!-- mask-size ---------------------------------------------------------------------------------------------------------------------------->

# mask-size

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

<!-- object-fit ------------------------------------------------------------------------------------------------------------------------------>

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

<!-- object-position ----------------------------------------------------------------------------------------------------------------------->

# object-position

расположит изображение в контейнере

```scss
 {
  object-position: center top;
  object-position: 100px 50px;
}
```

<!-- outline-offset ------------------------------------------------------------------------------------------------------------------------>

# outline-offset

отступ от обводки внешней границы

```scss
 {
  outline-offset: 4px;
  outline-offset: 0.6rem;
}
```

<!-- position -------------------------------------------------------------------------------------------------------------------------------->

# position

```scss
 {
  //
  position: static; //нормальное расположение
  position: relative; //позиционирует элементы относительно своей нормальной позиции, с возможностью наехать на другой элемент
  position: absolute; //вытаскивает элемент из нормального потока
  position: fixed; //остается на одном и том же месте
  position: sticky; // ведет себя как static пока не достигнет края окна во время прокрутки
}
```

<!-- rotate -------------------------------------------------------------------------------------------------------------------------------->

# rotate

Позволяет вращать 3-d объект

```scss
.rotate {
  //* Angle value */
  rotate: 90deg;
  rotate: 0.25turn;
  rotate: 1.57rad;

  /* x, y, or z axis name plus angle */
  rotate: x 90deg;
  rotate: y 0.25turn;
  rotate: z 1.57rad;

  /* Vector plus angle value */
  rotate: 1 1 1 90deg;
}
```

<!-- scroll-snap-type ----------------------------------------------------------------------------------------------------------------------->

# scroll-snap-type

определяет строгость привязки

```scss
.scroll-snap-type {
  scroll-snap-type: none;
  scroll-snap-type: x; // Прокрутка контейнера привязывается только по горизонтальной оси.
  scroll-snap-type: y; // Прокрутка контейнера привязывается только по вертикальной оси.
  scroll-snap-type: block; // Прокрутка контейнера привязывается только по блоковой оси.
  scroll-snap-type: inline; // Прокрутка контейнера привязывается только по строчной оси
  scroll-snap-type: both; // Прокрутка контейнера независимо привязывается только по обоим осям (потенциально может привязываться к разным элементам на разных осях).
  // mandatory proximity
  scroll-snap-type: x mandatory; // определяет обязательное смещение прокрутки браузера к ближайшей точке привязки
  scroll-snap-type: y proximity; // привязка может произойти , но не обязательно.
}
```

<!-- top-right-bottom-left------------------------------------------------------------------------------------------------------------------>

# top-right-bottom-left

Позиционирование для position:absolute | relative | sticky. Если заданы height: auto | 100% то будут учитываться оба

```scss
 {
  //
}
```

<!-- vertical-align ------------------------------------------------------------------------------------------------------------------------>

# vertical-align

Позволяет вертикально выравнять inline или inline-block элемент (нужно применять к элементу, который нужно выровнять) может использоваться в таблицах

```scss
 {
  vertical-align: baseline;
  vertical-align: sub;
  vertical-align: super;
  vertical-align: text-top;
  vertical-align: text-bottom;
  vertical-align: middle;
  vertical-align: top;
  vertical-align: bottom;
}
```

<!-- word-wrap --------------------------------------------------------------------------------------------------------------------------->

# word-break

```scss
 {
  word-wrap: "normal" | "break-word" | "inherit"; //перенос строки при переполнении
}
```
