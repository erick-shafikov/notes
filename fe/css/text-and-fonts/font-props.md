Некоторые понятия связанные с начертанием шрифтов

Глиф — конкретное визуальное начертание символа в шрифте. Один символ (Unicode code point) может иметь несколько глифов: например, буква «а» в прямом и курсивном начертании — разные глифы. Шрифт — это набор глифов.

Кегель (em, кегельная площадь) — размер виртуального прямоугольника, в который вписывается глиф. Исторически — высота металлического литера вместе с полями сверху и снизу. В цифровых шрифтах кегель задаёт систему координат: все метрики (ascender, descender, line-gap) выражаются в долях кегля. `font-size` в CSS задаёт именно размер кегля, а не высоту видимых букв — поэтому буквы разных гарнитур при одинаковом `font-size` выглядят разного размера.

Ascender — часть глифа, выступающая выше базовой линии за пределы x-height (высоты строчных букв без выносных элементов). Например, верхние палочки букв «b», «d», «h», «l». Метрика `ascender` в шрифте задаёт максимальную высоту таких элементов.

Descender — часть глифа, опускающаяся ниже базовой линии. Например, хвосты букв «g», «p», «q», «y». Метрика `descender` задаёт максимальную глубину таких элементов.

Baseline (базовая линия) — воображаемая линия, на которой «стоят» строчные буквы без descender-ов. Все остальные метрики отсчитываются от неё. В CSS свойство `vertical-align: baseline` выравнивает элементы именно по этой линии.

X-height — высота строчных букв без выносных элементов (эталон — буква «x»). CSS-единица `1ex = x-height` текущего шрифта. Влияет на поведение `vertical-align: middle`: браузер выравнивает по `baseline + x-height/2` родителя, а не по настоящему геометрическому центру.

Capital Height — высота прописных букв. CSS-единица `1cap`. Важна при выравнивании иконок рядом с текстом — иконку размером `1cap` можно точно выровнять по высоте заглавных букв.

content-area — вычисленная область содержимого (область к которой может применяться background). Высота = (ascender + descender) \* font-size / upm. Это не то, что видно на экране: padding и border строчного элемента расширяют фоновую область, но не меняют высоту content-area и не влияют на line-box. margin-top/bottom у строчных элементов не работают вовсе. Для замещаемых элементов (img, input, svg) и inline-block — padding, margin, border увеличивают height и влияют на line-box.

Каждый шрифт задает свой em-квадрат (upm - units per em, единица на кегельную площадь) то есть площадка где будет рисоваться. Использются относительные единицы от 1000 условных единиц что соответствует 1. При вычислении box-размеров inline элементов браузер выделяет достаточную площадь для того чтобы внутренние элементы поместились. Разницу можно увидеть, если в один контейнер (например p) добавить просто строки и рядом, как соседей, элементы span высоты будут различаться

line-box (контейнер строки) — каждая визуальная строка текста = один line-box. Высота line-box считается от самой верхней до самой нижней точки дочерних строчных элементов (по их virtual-area, а не content-area). Элемент `p` — это стопка line-box-ов.

Strut — браузер неявно вставляет в начало каждого line-box символ нулевой ширины с метриками **родительского** шрифта. Из-за этого если `p` имеет один font-family, а вложенный `span` — другой, их baseline расходятся и line-box оказывается выше, чем ожидается.

leading — каждый строчный элемент имеет две области:

- content-area (определяется метриками шрифта)
- virtual-area (line-height, используется для вычисления высоты line-box)

Разница между ними — интерлиньяж (leading). Делится поровну: половина добавляется сверху к content-area, половина снизу — поэтому content-area всегда центрирована внутри virtual-area. Если `line-height < content-area`, leading отрицательный и line-box визуально меньше своих дочерних элементов.

# font

font = font-style + font-variant + font-weight + font-stretch + font-size + line-height + font-family

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
  font-family:
    Gill Sans Extrabold,
    sans-serif;
  font-family: "Goudy Bookletter 1911", sans-serif;
}
```

Разновидности шрифтов по типам:

- serif - с засечками
- sans-serif - без засечек.
- monospace - в которых все символы имеют одинаковую ширину, обычно используются в листингах кода.
- cursive - имитирующие рукописный почерк, с плавными, соединенными штрихами.
- fantasy - предназначенные для декоративных целей.

# font-size

размер шрифта, стандартное значение у тега html - 16px.

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

- Нужно помнить что font-size не гарантирует размер ровно указанной величине в пикселях, зависит от настроек шрифта

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

# line-height

расстояние между строками или минимальное расстояние между блоками в блоке, берется от текущего шрифта. Это расстояние между началом и концом строчного блока, а не расстояние между базовыми линиями.

`line-height: normal` — не 1 и не 1.2. Вычисляется из метрик шрифта: `(ascender + descender + line-gap) / em`. Для разных гарнитур варьируется от 0.618 до 3.378. Задавать `line-height: 1` — плохая практика: у большинства шрифтов (95%) `normal > 1`, и line-height: 1 сделает virtual-area меньше content-area (отрицательный leading).

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

# vertical-align

→ [../block-model/5. position.md#vertical-align](../block-model/5.%20position.md#vertical-align)
