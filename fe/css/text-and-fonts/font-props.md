# font

font = font-style + font-variant + font-weight + font-stretch + font-size + line-height + font-family

# font-style

стиль начертания

```scss
.font-style {
  font-style: normal;
  font-style: italic; //курсив
  font-style: oblique; //курсив
}
```

# font-variant:

font-variant-alternates + font-variant-caps + font-variant-east-asian + font-variant-emoji + font-variant-ligatures + font-variant-numeric + font-variant-position

варианты написания разных шрифтов под разные языки если они предусмотрены шрифтом

## font-variant-alternates

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

## font-variant-numeric

управляет использованием альтернативных начертаний для цифр, дробей и порядковых числительных.

```scss
.font-variant-numeric{

font-variant-numeric: normal;
font-variant-numeric: ordinal; // 1st, 2nd, 3rd, 4t
font-variant-numeric: slashed-zero; // 0 с черточкой
font-variant-numeric: lining-nums; // заглавные цифры
font-variant-numeric: oldstyle-nums; // цифры уходят вниз строки
font-variant-numeric: proportional-nums; // разная ширина
font-variant-numeric: tabular-nums; // одиннаковая ширина
font-variant-numeric: diagonal-fractions; // дроби с косой чертой
font-variant-numeric: stacked-fractions; // дроби с горизонтальной чертой
font-variant-numeric: oldstyle-nums stacked-fractions;

font-variant-numeric: initial
font-variant-numeric: inherit
font-variant-numeric: unset
}
```

# font-weight

жирность

```scss
.font-weight {
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

# font-stretch

растягивает шрифт

```scss
.font-stretch {
  font-stretch: normal;
  font-stretch: ultra-condensed; //все значения от 62.5%
  font-stretch: extra-condensed;
  font-stretch: condensed;
  font-stretch: semi-condensed;
  font-stretch: semi-expanded;
  font-stretch: expanded;
  font-stretch: extra-expanded;
  font-stretch: ultra-expanded; //до 200%

  font-stretch: 50%;
  font-stretch: 100%;
  font-stretch: 200%;
}
```

# font-size

размер шрифта, стандартное значение у тега html - 16px

Возможные значения:

- 1mm (мм) = 3.8px (не используются)
- 1cm (см) = 38px (не используются)
- 1pt (типографский пункт) = 4/3 px (не используются)
- 1pc (типографская пика) = 16px (не используются)
- % - от родителя

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

## @font-feature-values

Применение для нескольких font-variant-alternates

```scss
/* Правило для "хорошего стиля" в Font One */
@font-feature-values Font One {
  @styleset {
    nice-style: 12;
  }
}

/* Правило для "хорошего стиля" в Font Two */
@font-feature-values Font Two {
  @styleset {
    nice-style: 4;
  }
}

…

/* Применение правила с единственным объявлением */
.nice-look {
  font-variant-alternates: styleset(nice-style);
}
```

# line-height

расстояние между строками или минимальное расстояние между блокам и в блоке, берется от текущего шрифта

```scss
.line-height {
  line-height: 1rem; //px | % | 1.5;
}
```

можно центрировать одну строку задав одинаковый height и line-height

```scss
.outer {
  height: 5em;
  line-height: 5em;
  border: 1px solid blue;
}
```

```html
<div class="outer">
  <span style="border:1px solid red">Текст</span>
</div>
```

# font-family

список из шрифтов, которые предустановлены либо определены [@font-face](#font-face)

```scss
 .font-family{
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

```scss
// перечисление нескольких не через запятую
.font-family {
  font-family: Gill Sans Extrabold, sans-serif;
  font-family: "Goudy Bookletter 1911", sans-serif;
}
```

Разновидности шрифтов по типам:

- serif - с засечками
- sans-serif - без засечек.
- monospace - в которых все символы имеют одинаковую ширину, обычно используются в листингах кода.
- cursive - имитирующие рукописный почерк, с плавными, соединенными штрихами.
- fantasy - предназначенные для декоративных целей.
