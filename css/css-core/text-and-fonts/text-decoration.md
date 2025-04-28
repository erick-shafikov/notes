# color

цвет текста

```scss
.color {
  color: red; //цвет текста
}
```

можно использовать свойство -webkit-text-fill-color

# font-palette

для взаимодействия с цветами в цветных шрифтах

## @font-palette-values

изменение цветов по умолчанию

```scss
@font-palette-values --identifier {
  font-family: Bixa;
}
.my-class {
  font-palette: --identifier;
}
```

# -webkit-text-stroke

webkit-text-stroke = -webkit-text-stroke-width and -webkit-text-stroke-color.

цвет обводки

```scss
.webkit-text-stroke {
  -webkit-text-stroke: 2px red;
}
```

# text-decoration:

свойства text-decoration = text-decoration-line + text-decoration-color + text-decoration-style + text-decoration-thickness, декорирование подчеркивания текста

## text-decoration-line

```scss
.text-decoration-line {
  //декорирование текста
  text-decoration-line: underline | overline | line-through | blink; //где находится линия
  text-decoration-line: underline overline; // может быть две
  text-decoration-line: overline underline line-through;

  // цвет знака ударения
  text-emphasis-color: currentColor;
}
```

## text-decoration-color

цвет подчеркивания

```scss
.text-decoration {
  // шорткат для text-decoration-line, text-decoration-style, ext-decoration-color
  text-decoration: line-through red wavy;
  text-decoration-color: red;
}
```

## text-decoration-style

```scss
.text-decoration-style {
  text-decoration-style: solid | double | dotted | dashed | wavy;
} //цвет линии
```

## text-decoration-thickness

ширина линии подчеркивания

```scss
.text-decoration-thickness {
  text-decoration-thickness: 0.1em;
  text-decoration-thickness: 3px;
}
```

## text-underline-offset (text-decoration)

text-underline-offset: px - позволяет определить расстояния от линии декоратора до текста при text-decoration

## text-underline-position (text-decoration)

при text-decoration

text-underline-position: auto | under - позволяет определить линия подчеркивания будет находит внизу всех элементов

## text-decoration-skip

при добавлении подчеркивания сделать сплошную линию, либо с прерыванием на буквы у,р,д

```scss
.text-decoration-skip-ink {
  text-decoration-skip-ink: auto | none;
}
```

# text-emphasis (верх):

Добавит элементы поверх текста, text-emphasis = text-emphasis-position + text-emphasis-style + text-emphasis-color.

```scss
.text-emphasis {
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

## text-emphasis-color - цвет элементов поверх текста

```scss
.text-emphasis-color {
  text-emphasis-color: #555;
  text-emphasis-color: blue;
  text-emphasis-color: rgb(90 200 160 / 80%);
}
```

## text-emphasis-position расположение элементов поверх текста

```scss
.text-emphasis-position {
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

## text-emphasis-style элемент вставки

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

# text-shadow

тень от текста

```scss
.text-shadow {
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

# text-transform

преобразует написание текста upper/lower-case и др

```scss
.text-transform {
  text-transform: none;
  text-transform: capitalize;
  text-transform: uppercase;
  text-transform: lowercase;
  text-transform: full-width; //выравнивание нестандартных шрифтов
  text-transform: full-size-kana; //ruby-текст аннотации
  text-transform: math-auto; //математический курсив
}
```

# initial-letter

размер и глубину для опущенных, приподнятых и утопленных начальных букв.

```scss
.initial-letter {
  initial-letter: normal;
  initial-letter: 3; //насколько шире
  initial-letter: 3 2; //на сколько шире на сколько вниз уходит
/
}
```

# user-select

Отвечает за возможность выделять текст

```scss
.user-select {
  user-select: none;
  user-select: auto;
  user-select: text; //Текст может быть выбран пользователем.
  user-select: contain; //Позволяет начать выбор внутри элемента; однако, выбор будет содержаться внутри границ данного элемента.
  user-select: all;
}
```

# font-size-adjust

позволяет регулировать lowercase и uppercase

```scss
.font-size-adjust {
  font-size-adjust: none;

  font-size-adjust: 0.5;
  font-size-adjust: from-font;

  font-size-adjust: ex-height 0.5;
  font-size-adjust: ch-width from-font;
}
```

# quotes

режим кавычек для тега q

```scss
.quotes {
  quotes: initial;
  quotes: "'" "'";
  quotes: "„" "“" "‚" "‘";
  quotes: "«" "»" "‹" "›";
}
```
