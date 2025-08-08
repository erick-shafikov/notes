<!-- Маски clip-path ------------------------------------------------------------------------------------------------------------------------->

# Маски clip-path:

## clip-path

Позволяет определить фигуру

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

<!-- mask ------------------------------------------------------------------------------------------------------------------------------------>

# mask

Маски подобны изображениям

краткая запись следующих свойств нужная для маскирования изображения:

mask = mask-clip + mask-composite + mask-image + mask-mode + mask-origin + mask-position + mask-repeat + mask-size

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

ресурс для маски, может быть:

- изображением (с прозрачным фоном)
- градиентом с использованием transparent

Можно использовать для маскирования изображений и текста

```scss
 {
  mask-image: url(masks.svg#mask1);

  /* <image> values */
  mask-image: linear-gradient(rgb(0 0 0 / 100%), transparent);
  mask-image: linear-gradient(#000, transparent);
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

```scss
.mask-type {
  mask-type: alpha; //использование в качестве маски alpha
  mask-type: luminance; //использование в качестве маски яркость
}
```

## mask-composite

определяет поведение при наложении маск

```scss
.mask-composite {
  mask-composite: add; //
  mask-composite: subtract;
  mask-composite: intersect;
  mask-composite: exclude;
}
```
