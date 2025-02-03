<!-- @charset------------------------------------------------------------------------------------------------------------------------------->

# @charset

Позволяет определить кодировку отличную от ascii

```scss
@charset "UTF-8";
@charset "iso-8859-15";
```

<!-- @color-profile -------------------------------------------------------------------------------------------------------------------------->

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

<!-- @document ------------------------------------------------------------------------------------------------------------------------------->

# @document

ограничение по стилю в зависимости от url

```scss
@document url("https://www.example.com/")
{
  h1 {
    color: green;
  }
}
```

- url(), который соответствует точному URL-адресу.
- url-prefix(), который совпадает, если URL-адрес документа начинается с указанного значения.
- domain(), который совпадает, если URL-адрес документа находится в предоставленном домене (или его субдомене).
- media-document(),с параметром видео, изображения, плагина или всего.
- regexp(), который совпадает, если URL-адрес документа сопоставляется с предоставленным регулярным выражением. Выражение должно соответствовать всему URL-адресу.

<!-- @import --------------------------------------------------------------------------------------------------------------------------------->

# @import

Позволяет импортировать стили, должно быть на верху фала, кроме @charset

```scss
//путь до файла
@import "url";
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

<!-- layer ----------------------------------------------------------------------------------------------------------------------------------->

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
//and
@media screen and (width: 600px) {
}
// для максимальных и минимальных значений
@media screen and (max-width: 600px) {
}
// для промежутков
@media (min-width: 30em) and (max-width: 50em) {
  //
}
// более сокращенный вариант
@media (30em <= width <= 50em) {
  //
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
  //
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

Позволяет определить стиль для разных пространств имен. Предназначен для стилизации svg, xml

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
