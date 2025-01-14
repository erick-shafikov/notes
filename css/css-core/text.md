# добавление шрифтов на сайт:

```html
<link
  href="http://fonts.googleapis.com/css?family=Open+Sans"
  rel="stylesheet"
  type="text/css"
/>
```

## @font-face

установка шрифтов с помощью

```scss
 {
  font-family: 'Name of font'; //даем название шрифту
  src: url(); //указываем место, где находится шрифт
  src: local(); //указываем место, где находится шрифт на устройстве пользователя

}

@font-face {
  font-family: "Open Sans";
  src: url("/fonts/OpenSans-Regular-webfont.woff2") format("woff2"), url("/fonts/OpenSans-Regular-webfont.woff") format("woff");
  // использование локальных шрифтов
  src: local("Helvetica Neue Bold"), local("HelveticaNeue-Bold"),
  // ------------------------------------------------------------------
  font-display: auto;
  font-display: block;
  font-display: swap;
  font-display: fallback;
  font-display: optional;
  // ------------------------------------------------------------------
  font-stretch: ultra-condensed;
  font-stretch: extra-condensed;
  font-stretch: condensed;
  font-stretch: semi-condensed;
  font-stretch: normal;
  font-stretch: semi-expanded;
  font-stretch: expanded;
  font-stretch: extra-expanded;
  font-stretch: ultra-expanded;
  font-stretch: 50%;
  font-stretch: 100%;
  font-stretch: 200%;
  // ------------------------------------------------------------------
  font-style: normal;
  font-style: italic;
  font-style: oblique;
  font-style: oblique 30deg;
  font-style: oblique 30deg 50deg;
  // ------------------------------------------------------------------
  font-weight: normal;
  font-weight: bold;
  font-weight: 400;
  /* Multiple Values */
  font-weight: normal bold;
  font-weight: 300 500;
}
```

использование

```scss
html {
  font-family: "myFont", "Bitstream Vera Serif", serif;
}
```

# font:

font = font-style + font-variant + font-weight + font-stretch + font-size + line-height + font-family

## font-style

стиль начертания

```scss
 {
  font-style: normal;
  font-style: italic; //курсив
  font-style: oblique; //курсив
}
```

## font-variant:

font-variant-alternates + font-variant-caps + font-variant-east-asian + font-variant-emoji + font-variant-ligatures + font-variant-numeric + font-variant-position

варианты написания разных шрифтов под разные языки если они предусмотрены шрифтом

### font-variant-alternates

управляет использованием альтернативных глифов

```scss
.font-variant-alternate {
  font-variant-alternates: stylistic(user-defined-ident);
  font-variant-alternates: styleset(user-defined-ident);
  font-variant-alternates: character-variant(user-defined-ident);
  font-variant-alternates: swash(user-defined-ident);
  font-variant-alternates: ornaments(user-defined-ident);
  font-variant-alternates: annotation(user-defined-ident);
  font-variant-alternates: swash(ident1) annotation(ident2);
}
```

## font-weight жирность

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

## font-stretch

растягивает шрифт

```scss
.font-stretch {
  font-stretch: normal;
  font-stretch: ultra-condensed; //62.5%
  font-stretch: extra-condensed;
  font-stretch: condensed;
  font-stretch: semi-condensed;
  font-stretch: semi-expanded;
  font-stretch: expanded;
  font-stretch: extra-expanded;
  font-stretch: ultra-expanded; //200%

  font-stretch: 50%;
  font-stretch: 100%;
  font-stretch: 200%;
}
```

## font-size

размер шрифта, стандартное значение у тега html - 16px

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

## line-height

расстояние между строками

```scss
 {
  line-height: "px", "%";
}
```

## font-family

список из шрифтов

```scss
 {
  // оба определения валидные
  font-family: Gill Sans Extrabold, sans-serif;
  font-family: "Goudy Bookletter 1911"//если название шрифта состоит из нескольких слов, то нужно заключать в кавычки

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

Разновидности шрифтов по типам:

- serif - с засечками
- sans-serif - без засечек.
- monospace - в которых все символы имеют одинаковую ширину, обычно используются в листингах кода.
- cursive - имитирующие рукописный почерк, с плавными, соединенными штрихами.
- fantasy - предназначенные для декоративных целей.

## -----------------------------------------------------

## font-feature-settings

если шрифты имеют доп настройки

```scss
.font-feature-settings {
  font-feature-settings: "smcp";
  font-feature-settings: "smcp" on;
  font-feature-settings: "swsh" 2;
  font-feature-settings: "smcp", "swsh" 2;
}
```

## font-kerning

расстояние между буквами

```scss
.font-kerning {
  font-kerning: auto;
  font-kerning: normal;
  font-kerning: none;
}
```

## font-language-override (-chrome, -safari, -ff)

переопределение очертания для других языков

## font-optical-sizing

значения: none | auto - оптимизация

## font-palette

для взаимодействия с цветами

## font-size-adjust

позволяет регулировать lowercase и uppercase

```scss
 {
  font-size-adjust: none;

  font-size-adjust: 0.5;
  font-size-adjust: from-font;

  font-size-adjust: ex-height 0.5;
  font-size-adjust: ch-width from-font;
}
```

## font-synthesis

font-synthesis = font-synthesis-weight + font-synthesis-style + font-synthesis-small-caps + font-synthesis-position

## font-variant:

font-variant-alternates
font-variant-caps
font-variant-east-asian
font-variant-emoji
font-variant-ligatures
font-variant-numeric
font-variant-position

## font-face (js)

[возможность управлять шрифтами через js](../../js/web-api/font-face.md)

# настройки расстояния

## word-spacing

```scss
 {
  word-spacing: "px", "%";
}
```

## letter-spacing

расстояние между буквами

```scss
 {
  letter-spacing: "px", "%";
}
```

## tab-size

размер символа табуляции

## text-indent

определяет размер отступа (пустого места) перед строкой в текстовом блоке.

## white-space

Свойство white-space управляет тем, как обрабатываются пробельные символы внутри элемента.

```scss
 {
  white-space: normal; //Последовательности пробелов объединяются в один пробел.
  white-space: nowrap; //не переносит строки (оборачивание текста) внутри текста.
  white-space: pre; //Последовательности пробелов сохраняются так, как они указаны в источнике.
  white-space: pre-wrap; //как и в pre + <br/>
  white-space: pre-line; //только <br />
  white-space: break-spaces;
}
```

## white-space-collapse

управляет тем, как сворачивается пустое пространство внутри элемента

<!--  -->

# расположение текста в контейнере

## text-align

CSS-свойство описывает, как линейное содержимое, наподобие текста, выравнивается в блоке его родительского элемента. text-align не контролирует выравнивание элементов самого блока, но только их линейное содержимое.

```scss
.text-align {
  text-align: left;
  text-align: right;
  text-align: center;
  text-align: justify;
  text-align: start;
  text-align: end;
  text-align: match-parent; //c учетом direction
  text-align: start end;
  text-align: "."; // до символа
  text-align: start ".";
  text-align: "." end;
}
```

## alignment-baseline (-ff)

Свойство CSS определяет определенную базовую линию, используемую для выравнивания текста блока и содержимого на уровне строки. Выравнивание базовой линии — это отношение между базовыми линиями нескольких объектов выравнивания в контексте выравнивания

```scss
.alignment-baseline {
  alignment-baseline: alphabetic;
  alignment-baseline: central;
  alignment-baseline: ideographic;
  alignment-baseline: mathematical;
  alignment-baseline: middle;
  alignment-baseline: text-bottom;
  alignment-baseline: text-top;

  /* Mapped values */
  alignment-baseline: text-before-edge; /* text-top */
  alignment-baseline: text-after-edge; /* text-bottom */
}
```

## dominant-baseline

Свойство CSS определяет определенную базовую линию, используемую для выравнивания текста и содержимого на уровне строки в блоке.

```scss
.dominant-baseline {
  dominant-baseline: alphabetic;
  dominant-baseline: central;
  dominant-baseline: hanging;
  dominant-baseline: ideographic;
  dominant-baseline: mathematical;
  dominant-baseline: middle;
  dominant-baseline: text-bottom;
  dominant-baseline: text-top;
}
```

<!-- настройка разрыва строк ----------------------------------------------------------------------------------------------------------------->

# декорирование текста

## color

цвет текста

```scss
 {
  color: red; //цвет текста
}
```

## text-decoration:

свойства text-decoration = text-decoration-line + text-decoration-color + text-decoration-style + text-decoration-thickness, декорирование подчеркивания текста

### text-decoration-line

```scss
 {
  //декорирование текста
  text-decoration-line: underline | overline | line-through | blink; //где находится линия
  text-decoration-line: underline overline; // может быть две
  text-decoration-line: overline underline line-through;

  // цвет знака ударения
  text-emphasis-color: currentColor;
}
```

### text-decoration-color

цвет подчеркивания

```scss
 {
  // шорткат для text-decoration-line, text-decoration-style, ext-decoration-color
  text-decoration: line-through red wavy;
  text-decoration-color: red;
}
```

### text-decoration-style

```scss
.text-decoration-style {
  text-decoration-style: solid | double | dotted | dashed | wavy;
} //цвет линии
```

### text-decoration-thickness

ширина линии подчеркивания

```scss
 {
  text-decoration-thickness: 0.1em;
  text-decoration-thickness: 3px;
}
```

### более тонкие настройки:

#### text-underline-offset

text-underline-offset: px - позволяет определить расстояния от линии декоратора до текста

#### text-underline-position

text-underline-position: auto | under - позволяет определить линия подчеркивания будет находит внизу всех элементов

#### text-decoration-skip

при добавлении подчеркивания сделать сплошную линию, либо с прерыванием на буквы у,р,д

```scss
 {
  text-decoration-skip-ink: auto | none;
}
```

## text-emphasis:

Добавит элементы поверх текста, text-emphasis = text-emphasis-position + text-emphasis-style + text-emphasis-color.

```scss
 {
  text-emphasis: "x";
  text-emphasis: "点";
  text-emphasis: "\25B2";
  text-emphasis: "*" #555;
  text-emphasis: "foo"; /* Should NOT use. It may be computed to or rendered as 'f' only */

  /* Keywords value */
  text-emphasis: filled;
  text-emphasis: open;
  text-emphasis: filled sesame;
  text-emphasis: open sesame;

  // возможные значения
  //  dot | circle | double-circle | triangle | sesame

  /* Keywords value combined with a color */
  text-emphasis: filled sesame #555;
}
```

### text-emphasis-color - цвет элементов поверх текста

```scss
 {
  text-emphasis-color: #555;
  text-emphasis-color: blue;
  text-emphasis-color: rgb(90 200 160 / 80%);
}
```

### text-emphasis-position расположение элементов поверх текста

```scss
text-emphasis-position. {
  text-emphasis-position: auto;

  /* Keyword values */
  text-emphasis-position: over;
  text-emphasis-position: under;

  text-emphasis-position: over right;
  text-emphasis-position: over left;
  text-emphasis-position: under right;
  text-emphasis-position: under left;

  text-emphasis-position: left over;
  text-emphasis-position: right over;
  text-emphasis-position: right under;
  text-emphasis-position: left under;
}
```

### text-emphasis-style элемент вставки

```scss
.text-emphasis-style {
  text-emphasis-style: "x";
  text-emphasis-style: "\25B2";
  text-emphasis-style: "*";

  /* Keyword values */
  text-emphasis-style: filled;
  text-emphasis-style: open;
  text-emphasis-style: dot;
  text-emphasis-style: circle;
  text-emphasis-style: double-circle;
  text-emphasis-style: triangle;
  text-emphasis-style: filled sesame;
  text-emphasis-style: open sesame;
}
```

## text-shadow

тень от текста

```scss
 {
  /* смещение-x | смещение-y | радиус-размытия | цвет */
  text-shadow: 1px 1px 2px black;

  /* цвет | смещение-x | смещение-y | радиус-размытия */
  text-shadow: #fc0 1px 0 10px;

  /* смещение-x | смещение-y | цвет */
  text-shadow: 5px 5px #558abb;

  /* цвет | смещение-x | смещение-y */
  text-shadow: white 2px 5px;

  /* смещение-x | смещение-y
  Используем значения по умолчанию для цвета и радиуса-размытия */
  text-shadow: 5px 10px;

  //множественные тени
  text-shadow: 1px 1px 1px red, 2px 2px 1px red;
}
```

## text-transform

преобразует написание текста upper/lower-case и др

```scss
 {
  text-transform: none;
  text-transform: capitalize;
  text-transform: uppercase;
  text-transform: lowercase;
  text-transform: full-width; //выравнивание нестандартных шрифтов
  text-transform: full-size-kana; //ruby-текст аннотации
  text-transform: math-auto; //математический курсив
}
```

## initial-letter

initial-letter: number (экспериментальное) стилизация первой буквы

## user-select

Отвечает за возможность выделять текст

```scss
.user-select {
  user-select: none;
  user-select: auto;
  user-select: text;
  user-select: contain;
  user-select: all;
}
```

<!-- Разрыв и перенос ------------------------------------------------------------------------------------------------------------------------>

# разрыв и перенос

## word-break

Где будет установлен перевод на новую строку

```scss
.word-break {
  word-break: normal;
  word-break: break-all;
  word-break: keep-all;
  word-break: break-word;
}
```

## text-wrap

перенос слов

```scss
.text-wrap {
  text-wrap: wrap; //обычный перенос при переполнение
  text-wrap: nowrap; //отмена переноса
  text-wrap: balance; //лучшее соотношение в плане длины строк
  text-wrap: pretty; // более медленный алгоритм wrap
  text-wrap: stable;
}
```

менее поддерживаемые свойство - text-wrap-mode, text-wrap-style

## overflow-wrap

разрыв сплошных строк при переносе

```scss
 {
  overflow-wrap: normal;
  overflow-wrap: break-word; //мягкий разрыв предусматривается
  overflow-wrap: anywhere; //мягкий разрыв не предусматривается
}
```

## hyphens

указывает, как следует переносить слова через дефис, когда текст переносится на несколько строк

```scss
 {
  hyphens: none;
  hyphens: manual;
  hyphens: auto;
  -moz-hyphens: auto;
  -ms-hyphens: auto;
  -webkit-hyphens: auto;
  //правильный разделитель слов (*)
  hyphens: auto;
}
```

## hyphenate-character

```scss
.hyphenate-character {
  hyphenate-character: <string>;
  hyphenate-character: auto;
}
```

```html
<dl>
  <dt><code>hyphenate-character: "="</code></dt>
  <dd id="string" lang="en">Superc&shy;alifragilisticexpialidocious</dd>
  <dt><code>hyphenate-character is not set</code></dt>
  <dd lang="en">Superc&shy;alifragilisticexpialidocious</dd>
</dl>
```

```scss
dd {
  width: 90px;
  border: 1px solid black;
  hyphens: auto;
}

dd#string {
  -webkit-hyphenate-character: "=";
  hyphenate-character: "=";
}
```

## hyphenate-limit-chars

(экс) для определения количества букв в переносе

## text-overflow

при переполнении текстом строки overflow: hidden; white-space: nowrap

```scss
 {
  // обрежет текст
  text-overflow: clip;
  // поставит троеточие (два значения для rtl)
  text-overflow: ellipsis ellipsis;
  text-overflow: ellipsis " [..]";
  text-overflow: ellipsis "[..] ";
}
```

Так же могут помочь символ `&shy` `<wbr>​`;

## text-align-last

Как выравнивается последняя строка в блоке или строка, идущая сразу перед принудительным разрывом строки.

```scss
.text-align-last {
  text-align-last: auto;
  text-align-last: start;
  text-align-last: end;
  text-align-last: left;
  text-align-last: right;
  text-align-last: center;
  text-align-last: justify;
}
```

<!-- Направление письма ---------------------------------------------------------------------------------------------------------------------->

# направление письма

Блочная модель так же предусматривает направление текста

## writing-mode

устанавливает горизонтальное или вертикальное положение текста также как и направление блока

```scss
.writing-mode {
  writing-mode: horizontal-tb; // поток - сверху вниз, предложения - слева направо
  writing-mode: vertical-rl; // поток - справа налево, предложения - вертикально
  writing-mode: vertical-lr; // поток - слева направо, предложения - вертикально
}
```

## direction

принимает два значения ltr и rtl

## text-orientation

позволяет распределить символы в вертикальном и горизонтальном направлениях

```scss
.text-orientation {
  text-orientation: mixed;
  text-orientation: upright;
  text-orientation: sideways-right;
  text-orientation: sideways;
  text-orientation: use-glyph-orientation;
}
```

## block-size

Свойство позволяет записать height и width в одно свойство с учетом режима письма writing-mode.

```scss
.block-size {
  block-size: 300px;
  block-size: 25em;

  block-size: 75%;

  block-size: 25em border-box;
  block-size: 75% content-box;
  block-size: max-content;
  block-size: min-content;
  block-size: available;
  block-size: fit-content;
  block-size: auto;
}
```

## inset

позволяет определить top|bottom или right|left в зависимости от rtl

### inset-block

позволяет определить top|bottom или right|left в зависимости от rtl более точные свойства для управление расположением:

- - - inset-block-end
- - - inset-block-start
- - inset-inline аналогично и inset-block только представляет горизонтальную ориентацию
- - - inset-inline-end
- - - inset-inline-start

## text-combine-upright

учет чисел при написании в иероглифах all - все числа будут упакованы в размер одного символа

## line-break

перенос китайского и японского

<!-- ссылки ---------------------------------------------------------------------------------------------------------------------------------->

# ссылки

Состояния :

- :link - не посещенная
- :visited - посещенная
- :hover
- :active - при клике

Для стилизации используются:

## color

цвет ссылки

стилизация курсора

## cursor

Определяет тип курсора

[cursor](./user-elements.md#cursor)

<!-- BPs ------------------------------------------------------------------------------------------------------------------------------------->

# BPs

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

## BP. Текст, который залит фоном:

обрезка фона под текст

```css
.text-clip {
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
```

## BP. Улучшения по работе со шрифтами:

- использовать woff2
- пред загрузка
- фрагментация
- использование font-display
- локальное хранение шрифта

https://fonts.google.com/ - для поиска шрифтов
