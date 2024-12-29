<!-- Текстовый контент: текст и ссылки ---------------------------------------------------------------------------------------------------------------------------------->

Улучшения по работе со шрифтами:

- использовать woff2
- пред загрузка
- фрагментация
- использование font-display
- локальное хранение шрифта

# Текстовый контент: текст и ссылки

https://fonts.google.com/ - для поиска шрифтов

- font = font-style + font-variant + font-weight + font-stretch + font-size + line-height + font-family свойства шрифта

- - font-family список из шрифтов

```scss
 {
  // оба определения валидные
  font-family: Gill Sans Extrabold, sans-serif;
  font-family: "Goudy Bookletter 1911", sans-serif;

  /* Только общие семейства */
  font-family: serif; //со штрихами
  font-family: sans-serif; //гладкие
  font-family: monospace; //одинаковая ширина
  font-family: cursive; //рукопись
  font-family: fantasy; //декор-ые
  font-family: system-ui; //из системы
  font-family: emoji; //
  font-family: math; //
  font-family: fangsong; //китайский
}
```

- - font-size размер шрифта

```scss
.font-size {
  /* значения в <абсолютных размерах> */
  font-size: xx-small;
  font-size: x-small;
  font-size: small;
  font-size: medium;
  font-size: large;
  font-size: x-large;
  font-size: xx-large;
  /* значения в <относительных размерах> */
  font-size: larger;
  font-size: smaller;
  font-size: 12px;
  font-size: 0.8em;
  font-size: 80%;
}
```

```scss
body {
  // Масштабирование с помощью font-size
  font-size: 62.5%; /* font-size 1em = 10px on default browser settings */
}

span {
  font-size: 1.6em; /* 1.6em = 16px */
}
```

- - font-style стиль начертания

```scss
 {
  font-style: normal;
  font-style: italic; //курсив
  font-style: oblique; //курсив
}
```

- - font-weight жирность

```scss
 {
  /font-weight: normal;
  font-weight: bold;

  /* Relative to the parent */
  font-weight: lighter;
  font-weight: bolder;

  font-weight: 100;
  font-weight: 200;
  font-weight: 300;
  font-weight: 400;
  font-weight: 500;
  font-weight: 600;
  font-weight: 700;
  font-weight: 800;
  font-weight: 900;
}
```

- - [возможность управлять шрифтами через js](../../js/web-api/font-face.md)
- - расширенные настройки шрифтов:
- - - font-feature-settings если шрифты имеют доп настройки
- - - font-kerning если шрифты имеют доп настройки
- - - font-optical-sizing: none | auto оптимизация
- - - font-palette для взаимодействия с цветами
- - - font-size-adjust позволяет регулировать lowercase и uppercase
- - - font-stretch растягивает шрифт
- - - font-synthesis = font-synthesis-weight + font-synthesis-style + font-synthesis-small-caps + font-synthesis-position
- - - font-synthesis-small-caps
- - - font-variant = font-variant-alternates + font-variant-caps + font-variant-east-asian + font-variant-emoji + font-variant-ligatures + font-variant-numeric + font-variant-position варианты написания разных шрифтов под разные языки если они предусмотрены шрифтом
- - - font-variation-settings предоставляет низкоуровневый вариант управления шрифтом
- [стиль строки]
- - word-spacing расстояние между словами

```scss
 {
  word-spacing: "px", "%";
}
```

- - letter-spacing расстояние между буквами

```scss
 {
  letter-spacing: "px", "%";
}
```

- - line-height расстояние между строками

```scss
 {
  line-height: "px", "%";
}
```

- - line-break перенос китайского и японского
- - text-align контроль расположения текста
- - text-align-last как будет выравнен текст в последней строке или перед разрывом
- - tab-size - размер символа табуляции
- - [color - цвет текста](./css-props.md#color)
- - text-indent определяет размер отступа (пустого места) перед строкой в текстовом блоке.

[свойства text-decoration = text-decoration-line + text-decoration-color + text-decoration-style + text-decoration-thickness, декорирование подчеркивания текста](./css-props.md/#text-decoration)

- - [цвет подчеркивания](./css-props.md/#text-decoration-color)
- - text-underline-offset: px - позволяет определить расстояния от линии декоратора до текста
- - text-underline-position: auto | under - позволяет определить линия подчеркивания будет находит внизу всех элементов
- - text-decoration-skip подчеркивание и буквы у,д, р с хвостом внизу
- - text-decoration-skip-ink: none | auto | all - наложение линии подчеркивания на буквы с нижней частью
- - [text-decoration-thickness ширина линии подчеркивания](./css-props.md/#text-decoration-thickness)

[text-emphasis Добавит элементы поверх текста](./css-props.md/#text-emphasis)

- - [text-emphasis-color - цвет элементов поверх текста](./css-props.md/#text-emphasis-color)
- - [text-emphasis-position расположение элементов поверх текста](./css-props.md/#text-emphasis-position)
- - [text-emphasis-style элемент вставки](./css-props.md/#text-emphasis-style)

Другие декораторы текста:

- [тень от текста](./css-props.md/#text-shadow)
- [text-transform преобразует написание текста upper/lower-case и др](./css-props.md/#text-transform)
- initial-letter: number (экспериментальное) стилизация первой буквы

- Текст, который залит фоном:
- - [обрезка фона под текст](./css-props.md#webkit-background-clip)
- - [заливка](./css-props.md#webkit-text-fill-color)
- [word-break - перенос слов c учетом языковых особенностей](./css-props.md#word-break)
- [text-wrap перенос слов](./css-props.md#text-wrap)
- - менее поддерживаемые свойство - text-wrap-mode, text-wrap-style
- [стилизация q quotes: "„" "“" "‚" "‘"; принимает закрывающие и открывающие кавычки]

word-break разрыв строк и перенос:

- [overflow-wrap разрыв сплошных строк при переносе](./css-props.md/#overflow-wrap)
- [разделение слов при переносе](./css-props.md#hyphens)
- [символ для разделения слова](./css-props.md/#hyphenate-character)
- hyphenate-limit-chars (экс) для определения количества букв в переносе
- [white-space - управление пробельными при переносе и пробельными символами](./css-props.md#white-space)
- white-space-collapse управляет тем, как сворачивается пустое пространство внутри элемента
- [text-overflow при переполнении текстом строки overflow: hidden; white-space: nowrap](./css-props.md/#text-overflow)

Так же могут помочь символ `&shy` `<wbr>​`;

Текст при выделении:

- [user-select - настройка выделение текста](./css-props.md/#user-select)

Направление письма:

- [writing-mode изменить направление текста](./css-props.md#writing-mode)

## Добавление шрифтов на сайт:

```html
<link
  href="http://fonts.googleapis.com/css?family=Open+Sans"
  rel="stylesheet"
  type="text/css"
/>
```

- [установка шрифтов с помощью @font-face](./at-rules.md#font-face)

использование

```scss
html {
  font-family: "myFont", "Bitstream Vera Serif", serif;
}
```

## ссылки

Состояния :

- :link - не посещенная
- :visited - посещенная
- :hover
- :active - при клике

Для стилизации используются:

- [цвет ссылки](./css-props.md#color)
- [стилизация курсора](./css-props.md#cursor)

## BP. Масштабирование всего проекта c помощью font-size

```css
html {
  /* 10px/16px = 0.625 - теперь 1 rem 10px */
  font-size: 62.5%;
}

/* базовый шрифт – 16px, для удобства верстки удобно отталкиваться от 10px, моно сделать с помощью rem – условная единица от базового шрифта */

body {
  /* 30 px – задаем все размеры в rem */
  padding: 3rem;
}
```

## BP. иконка в конец ссылки

```scss
a[href*="http"] {
  // иконка будет отодвинута в правый край
  background: url("external-link-52.png") no-repeat 100% 0;
  background-size: 16px 16px;
  // padding отодвинет иконку от последней буквы
  padding-right: 19px;
}
```

## BP. ссылки кнопки

```scss
a {
  // отменяем все стилизации
  outline: none;
  text-decoration: none;
  // меняем блочность
  display: inline-block;
  // в примере 5 кнопок
  width: 19.5%;
  // с учетом того, что в примере 5 кнопок
  margin-right: 0.625%;
  text-align: center;
  line-height: 3;
  color: black;
}

a:link,
a:visited,
a:focus {
  background: yellow;
}

a:hover {
  background: orange;
}

a:active {
  background: red;
  color: white;
}
```
