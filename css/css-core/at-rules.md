<!-- @charset------------------------------------------------------------------------------------------------------------------------------->

# @charset

Позволяет определить кодировку отличную от ascii

```scss
@charset "UTF-8";
@charset "iso-8859-15";
```

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# @color-profile

Определяет цветовой профиль

```scss
// имя --swop5c
@color-profile --swop5c {
  src: url("https://example.org/SWOP2006_Coated5v2.icc");
}

.header {
  background-color: color(--swop5c 0% 70% 20% 0%);
}
```

<!-- @container ------------------------------------------------------------------------------------------------------------------------------>

# @container

Позволяет настроить медиа запросы относительно элемента, а не vp. Задает контекст ограничения

```scss
// настройка для контейнера относительного которого будут производится измерения
.container {
  container-type: size; //size - измеряет как как inline или блочный
  container-type: inline-size; //inline-size - как inline
  container-type: normal; //Отключает запросы на размеры контейнера
}
```

```html
<!-- контейнер -->
<div class="post">
  <div class="card">
    <h2>Card title</h2>
    <p>Card content</p>
  </div>
</div>
```

```scss
.post {
  container-type: inline-size;
}
// контент .card и h2 будет изменять font-size при min-width: 700px
@container (min-width: 700px) {
  .card h2 {
    font-size: 2em;
  }
}
```

для множественных контейнеров, именованные контейнеры

```scss
.post {
  container-type: inline-size;
  // название контейнера
  container-name: sidebar;
  // сокращенное свойство
  container: sidebar / inline-size;
}
// использование
@container sidebar (min-width: 700px) {
  .card {
    font-size: 2em;
  }
}
```

доступны новые единицы измерения

- cqw: 1% от ширины контейнера
- cqh: 1% от высоты
- cqi: 1% от inline размера
- cqb: 1% от блочного размера
- cqmin: минимум от cqi и cqb
- cqmax: максимум от cqi и cqb

```scss
// использование
@container (min-width: 700px) {
  .card h2 {
    font-size: max(1.5em, 1.23em + 2cqi);
  }
}
```

Доступные свойства для условия: aspect-ratio, block-size, height, inline-size, orientation, width

вложенные

```scss
@container summary (min-width: 400px) {
  @container (min-width: 800px) {
    /* <stylesheet> */
  }
}
```

## Container style queries

c помощью функции style можно ссылать на стиль контейнера

```scss
@container style(<style-feature>),
    not style(<style-feature>),
    style(<style-feature>) and style(<style-feature>),
    style(<style-feature>) or style(<style-feature>) {
  /* <stylesheet> */
}

// пример

@container style(--themeBackground),
    not style(background-color: red),
    style(color: green) and style(background-color: transparent),
    style(--themeColor: blue) or style(--themeColor: purple) {
  /* <stylesheet> */

  //--themeColor: blue - незарегистрированное пользовательское свойство
}
```

<!-- @counter-style ------------------------------------------------------------------------------------------------------------------------>

# @counter-style

```scss
@counter-style symbols-example {
  system: cyclic | numeric | ....; //https://developer.mozilla.org/en-US/docs/Web/CSS/@counter-style/system
  symbols: A "1" "\24B7"D E; //ряд символов на повторение
  additive-symbols: 1000 M, 900 CM, 500 D, 400 CD, 100 C, 90 XC, 50 L, 40 XL,
  symbols: url(gold-medal.svg) url(silver-medal.svg) url(bronze-medal.svg); //могут быть и изображения
    10 X, 9 IX, 5 V, 4 IV, 1 I; //позволяет задать ряд с системой исчисления
  negative: "--"; // задать элементы, если они начинаются с отрицательного индекса атрибут start < 0
  prefix: "»";
  suffix: "";
  range: 2 4, 7 9; //на какие по счету элементы будет применяться
  pad: 3 "0";
  speak-as: auto |...;
  fallback: lower-alpha; //альтернативный
}

.items {
  list-style: symbols-example;
}
```

# @import

Позволяет импортировать стили, должно быть на верху фала, кроме @charset

```scss
//путь до файла
@import url;
@import "custom.css";
@import url("chrome://communicator/skin/");
//для разных устройств
@import url("fineprint.css") print;
@import url("bluish.css") print, screen;
@import "common.css" screen;
@import url("landscape.css") screen and (orientation: landscape);
//с учетом медиа запросов
@import url("gridy.css") supports(display: grid) screen and (max-width: 400px);
@import url("flexy.css") supports((not (display: grid)) and (display: flex)) screen
  and (max-width: 400px);
// supports
@import url("whatever.css") supports((selector(h2 > p)) and
    (font-tech(color-COLRv1)));
// c использованием layer
@import "theme.css" layer(utilities);
```

Пример с layer

```scss
@import url(headings.css) layer(default);
@import url(links.css) layer(default);

@layer default {
  audio[controls] {
    display: block;
  }
}
```

<!--@keyframes-------------------------------------------------------------------------------------------------------------------------------->

# @keyframes

Позволяет создать опорные точки анимации

свойства с !important будут проигнорированы

```scss
@keyframes slideIn {
  from {
    // 0%
    transform: translateX(0%);
  }

  to {
    // 100%
    transform: translateX(100%);
  }
}
// в процентах
@keyframes identifier {
  0% {
    top: 0;
    left: 0;
  }
  30% {
    top: 50px;
  }
  68%,
  72% {
    left: 50px;
  }
  100% {
    top: 100px;
    left: 100%;
  }
}
```

!important в keyframe будет игнорировано

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# @layer

для использования каскадных слоев

```scss
// объявляем (может быть несколько слоев)
@layer module, state;

// именованные слои
@layer state {
  .alert {
    background-color: brown;
  }
  p {
    border: medium solid limegreen;
  }
}

@layer module {
  .alert {
    border: medium solid violet;
    background-color: yellow;
    color: white;
  }
}

// анонимный
@layer {
  p {
    margin-block: 1rem;
  }
}
// импорт слоя из другого файла
@import "theme.css" layer(utilities);

// вложенные
@layer framework {
  @layer layout {
  }
}

// добавление дополнительных
@layer framework.layout {
  p {
    margin-block: 1rem;
  }
}
```

Импорт слоев из других файлов

```css
/* простые импорты */
import url("components-lib.css") layer(components);
@import url("dialog.css") layer(components.dialog);
@import url("marketing.css") layer();

/* с учетом медиа запросов */
@import url("ruby-narrow.css") layer(international) supports(display: ruby) and
  (width < 32rem);
@import url("ruby-wide.css") layer(international) supports(display: ruby) and
  (width >= 32rem);
```

<!-- @media ---------------------------------------------------------------------------------------------------------------------------------->

# @media

общий синтаксис

```scss
@media media-type and (media-feature-rule) {
  /* CSS rules go here */
}
```

Значения для media-type:

- all
- print
- screen
- speech

```scss
@media all {
  // для всех
}
@media (orientation: landscape) {
  // расположение
}
@media print {
  // расположение
}
@media screen {
  // расположение
}
```

Свойства на которые можно определить внутри скобок в медиа-запросе:

- width, height, aspect-ratio, orientation, resolution, scan, grid, update-frequency, overflow-block, overflow-inline, color, color-index, display-mode, monochrome, inverted-colors, pointer, hover, any-pointer, any-hover, scripting, device-width, device-height, device-aspect-ratio, -webkit-device-pixel-ratio, -webkit-transform-3d, -webkit-transform-2d, -webkit-transition, -webkit-animation

```scss
// доп функции --------------------------------------------------------
// and
//для конкретного значения
@media screen and (width: 600px) {
}
// для максимальных и минимальных значений
@media screen and (max-width: 600px) {
}
// для промежутков
@media (min-width: 30em) and (max-width: 50em) {
  /* … */
}
// более сокращенный вариант
@media (30em <= width <= 50em) {
  /* … */
}

// может ли пользоваться наведением
@media (hover: hover) {
}
// логическая комбинация или &
@media screen and (min-width: 600px) and (orientation: landscape) {
}
// логическая комбинация и
@media screen and (min-width: 600px), screen and (orientation: landscape) {
  body {
    color: blue;
  }
}
// логическая комбинация не
@media not (orientation: landscape) {
}
// логические комбинации
@media (not (width < 600px)) and (not (width > 1000px)) {
}
```

Позволяет настроить стили под определенные характеристики устройства

```scss
@media only screen and (min-resolution: 192dpi) and (min-width: 37.5em),
  only screen and (-webkit-min-device-pixel-ratio: 2) and (min-width: 37.5em),
  only screen and (max-width: 125em) {
  //...css правила
}
```

## @media(prefers-color-scheme)

Позволяет настроить стили для разных цветовых тем устройства

```scss
@media (prefers-color-scheme: dark | light) {
}
```

## @media(prefers-contrast)

Режим контраста

```scss
@media (prefers-contrast: no-preference | more | less | custom) {
  // без настроек
}
```

## @media(forced-colors)

Режим контраста

```scss
@media (forced-colors: none | active) {
  // без настроек
}
```

```scss
// для устройств ввода с мышью
@media (hover: hover) { ... }
// цветной экран
@media (color) { ... }
```

для печати

```html
<link href="/path/to/print.css" media="print" rel="stylesheet" />
```

<!-- @namespace ---------------------------------------------------------------------------------------------------------------------------->

# @namespace

Позволяет определить стиль для разных пространств имен. Предназначен для стилизации svg

```scss
@namespace url(http://www.w3.org/1999/xhtml);
@namespace svg url(http://www.w3.org/2000/svg);

/* This matches all XHTML <a> elements, as XHTML is the default unprefixed namespace */
a {
}

/* This matches all SVG <a> elements */
svg|a {
}

/* This matches both XHTML and SVG <a> elements */
*|a {
}
```

<!-- @page --------------------------------------------------------------------------------------------------------------------------------->

# @page

Позволяет определить стиль при печати страницы, изменить можно только margin, orphans, widows, и разрывы страницы документа.

```scss
@page {
  margin: 1cm;
}

@page :first {
  margin: 2cm;
}
```

Управление может происходить с помощью псевдоклассов :blank, :first, :left, :right

<!-- @position-try (якоря) ----------------------------------------------------------------------------------------------------------------------->

# @position-try (якоря)

Позволяет расположить якорь. Нет в сафари и в ff

```html
<div class="anchor">⚓︎</div>

<div class="infobox">
  <p>This is an information box.</p>
</div>
```

```scss
.anchor {
  anchor-name: --myAnchor;
  position: absolute;
  top: 100px;
  left: 350px;
}

@position-try --custom-left {
  inset-area: left;
  width: 100px;
  margin: 0 10px 0 0;
}

@position-try --custom-bottom {
  top: anchor(bottom);
  justify-self: anchor-center;
  margin: 10px 0 0 0;
  inset-area: none;
}

@position-try --custom-right {
  left: calc(anchor(right) + 10px);
  align-self: anchor-center;
  width: 100px;
  inset-area: none;
}

@position-try --custom-bottom-right {
  inset-area: bottom right;
  margin: 10px 0 0 10px;
}

.infobox {
  position: fixed;
  position-anchor: --myAnchor;
  inset-area: top;
  width: 200px;
  margin: 0 0 10px 0;
  position-try-fallbacks: --custom-left, --custom-bottom, --custom-right,
    --custom-bottom-right;
}
```

<!-- @property  ------------------------------------------------------------------------------------------------------------------------------------>

# @property

Позволяет создавать переисользуемые переменные с проверкой синтаксиса

```html
<div class="container">
  <div class="item one">Item one</div>
  <div class="item two">Item two</div>
  <div class="item three">Item three</div>
</div>
```

```scss
@property --item-size {
  syntax: "<percentage>";
  inherits: true;
  initial-value: 40%;
}
```

```js
window.CSS.registerProperty({
  name: "--item-color",
  syntax: "<color>",
  inherits: false,
  initialValue: "aqua",
});
```

```scss
.container {
  display: flex;
  height: 200px;
  border: 1px dashed black;

  /* set custom property values on parent */
  --item-size: 20%;
  --item-color: orange;
}

/* use custom properties to set item size and background color */
.item {
  width: var(--item-size);
  height: var(--item-size);
  background-color: var(--item-color);
}

/* set custom property values on element itself */
.two {
  --item-size: initial;
  --item-color: inherit;
}

.three {
  /* invalid values */
  --item-size: 1000px;
  --item-color: xyz;
}
```

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# @scope

Позволяет определить стили для поддеревьев

```scss
 @scope (scope root) to (scope limit) {
  rulesets
}
```

Может использоваться вместе с псевдо классом scope

<!-- @starting-style  ---------------------------------------------------------------------------------------------------------------------->

# @starting-style

Позволяет определить стили для начальных стадий анимации (полезно при display: none)

```scss
@starting-style {
  //стили
}
```

<!-- @supports ------------------------------------------------------------------------------------------------------------------------------->

# @supports

Позволяет проверить поддержку свойства в браузере

```scss
// проверка поддержки clip-path
@supports ((clip-path: polygon(0 0)) or (webkit-clip-path: polygon(0 0))) {
  -webkit-clip-path: polygon(0 0, 100% 0, 100% 75vh, 0 100%);
  clip-path: polygon(0 0, 100% 0, 100% 75vh, 0 100%);
  height: 95vh;
}
```

```scss
// использование импорта относительно поддерживаемого свойства supports
@import `/css/styles.css` supports(color: AccentColor);

// отсутствие поддержки
@supports not (property: value) {
  CSS rules to apply
}
// тест нескольких
@supports (property1: value) and (property2: value) {
  CSS rules to apply
}
// тест одной из двух
@supports (property1: value) or (property2: value) {
  CSS rules to apply
}
```

Дополнительные функции

```scss
// пример теста селектора также может быть и font-tech(), font-format() вместо selector()
@import `/css/webkitShadowStyles.css` supports(selector(::-webkit-inner-spin-button));
```

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# @view-transition

Нет в ff и сафари

```scss
@view-transition {
  navigation: auto;
  navigation: none; // Документ не будет подвергнут переходу вида.
}
```
