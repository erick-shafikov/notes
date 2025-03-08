<!-- Блочная модель --------------------------------------------------------------------------------------------------------------->

Элемент может быть блочным или строчным. Поток - это расположение элементов в документе. Что бы выкинуть элемент из потока из контекста форматирования - float, position: absolute, корневой элемент (html)
Контексты форматирования:

- block formatting context, BFC, располагаются по вертикали начиная с верху
- inline formatting context
- flex formatting context

Контекст форматирования formatting context - то как формируется поток, состоит из других потоков

Кроме корневого элемента html новый БКФ создаётся в следующих случаях:

- плавающие элементы (float: left или float: right);
- абсолютно позиционированные элементы (position: absolute, position: fixed или position: sticky);
- элементы с display: inline-block;
- ячейки табицы или элементы с display: table-cell, включая анонимные - ячейки таблицы, которые создаются, когда используются свойства display: table-\*;
- заголовки таблицы или элементы с display: table-caption;
- блочные элементы, когда значение свойства overflow отлично от visible;
- элементы с display: flow-root или display: flow-root list-item;
- элементы с contain: layout, content, или strict
- флекс-элементы;
- грид-элементы;
- контейнеры мультиколонок;
- элементы с column-span в значении all.

# display

определяет блочность/строчность элемента

```scss
.display {
  display: block;
  display: inline;
  display: run-in; //Если соседний элемент определён как display: run-in, тогда бокс является блоковым боксом, run-in бокс становится первым строковым (inline) боксом блокового бокса, который следует за ним.

  display: flow;
  display: flow-root; //устанавливает новый
  display: table;
  display: flex;
  display: grid;
  display: ruby; //модель форматирования Ruby

  display: block flow;
  display: inline table;
  display: flex run-in;

  // списковые
  display: list-item;
  display: list-item block;
  display: list-item inline;
  display: list-item flow;
  display: list-item flow-root;
  display: list-item block flow;
  display: list-item block flow-root;
  display: flow list-item block;

  // табличные
  display: table-row-group;
  display: table-header-group;
  display: table-footer-group;
  display: table-row;
  display: table-cell;
  display: table-column-group;
  display: table-column;
  display: table-caption;
  display: ruby-base;
  display: ruby-text;
  display: ruby-base-container;
  display: ruby-text-container;

  display: contents; //создаст псевдо-контейнер по своим дочерним элементам (не будет доступен, но будет в dom)
  display: none; //удаляем из дерева

  display: inline-block;
  display: inline-table;
  display: inline-flex;
  display: inline-grid;
}
```

- блочные боксы – прямоугольные области на странице, начинаются с новой строки, занимают всю доступную ширину, к ним применимы свойства width, height, элементы вокруг будет отодвинуты. Нужны для формирования структуры страницы. Занимает 100% ширины и высоту по содержимому. Если даже задать двум блоками идущим подряд ширину в 40% то они все равно расположатся друг под другом
- Строчные боксы – фрагменты текста span, a, strong, em, time у них нет переноса строки, ширина и высота зависят от содержимого, размеры задать нельзя за исключением элементов area и img. Не будут начинаться с новой строки, width, height недоступны, отступы не будут отодвигать другие элементы. Высота определяется по самому высокому элементу
- можно менять блочность/строчность через display

# Размеры:

## box-sizing

определяет как вычисляется величина контейнера.

- если задать ширину и высоту элементу, она будет применена для контента без учета рамок и отступа от рамок

```scss
 {
  //размеры буз учета рамок, стандартное поведение при отступах и рамках реальная ширина будет больше
  box-sizing: content-box;
  //будет учитывать размеры отступов
  box-sizing: content-box;
  //ужмется по контейнеру
  box-sizing: border-box;
}
```

```scss
div {
  width: 160px;
  height: 80px;
  padding: 20px;
  border: 8px solid red;
}

.content-box {
  box-sizing: content-box;
  /* Total width: 160px + (2 * 20px) + (2 * 8px) = 216px
     Total height: 80px + (2 * 20px) + (2 * 8px) = 136px
     Content box width: 160px
     Content box height: 80px */
}

.border-box {
  box-sizing: border-box;
  /* Total width: 160px
     Total height: 80px
     Content box width: 160px - (2 * 20px) - (2 * 8px) = 104px
     Content box height: 80px - (2 * 20px) - (2 * 8px) = 24px */
}
```

## width

```scss
 {
  // Ширина - фиксированная величина.
  width: 3.5em;
  width: anchor-size(width);
  width: calc(anchor-size(--myAnchor self-block, 250px) + 2em);

  width: 75%; // Ширина в процентах - размер относительно ширины родительского блока.

  width: none;
  width: max-content; //сожмет текстовой контент до размера самого МАЛЕНЬКОГО слова, остальные перенесет
  width: min-content; //сожмет текстовой контент до размера самого БОЛЬШОГО слова, остальные перенесет
  width: fit-content; //поле будет использовать доступное пространство, но не более max-content
  width: fit-content(20em); // min(maximum size, max(minimum size, argument))
}
```

## height

```scss
 {
  // если в процентах, то от контейнера
  height: 120px;
  height: 10em;
  height: 100vh;
  height: anchor-size(height);
  height: anchor-size(--myAnchor self-block, 250px);
  height: clamp(200px, anchor-size(width));

  /* <percentage> value */
  height: 75%;

  /* Keyword values */
  height: max-content;
  height: min-content;
  height: fit-content;
  height: fit-content(20em);
  height: auto;
  height: minmax(min-content, anchor-size(width));
  height: stretch;
}
```

для того что бы задать размеры отталкиваясь от минимальных и максимальных значений

- min-width, min-height, max-width, max-height – нужны для определения высоты контентных элементов, которые могут вывалиться
  max-width переопределяет width, но min-width переопределяет max-width. Свойства с учетом письма:
- - max-block-size
- - max-inline-size
- - min-block-size
- - min-inline-size

## aspect-ratio

позволяет настроить пропорции контейнера

```scss
.aspect-ratio {
  aspect-ratio: 1 / 1;
  aspect-ratio: 1;

  /* второе значение - запасное, если допустим изображение не загрузилось */
  aspect-ratio: auto 3/4;
  aspect-ratio: 9/6 auto;
}
```

## ориентация письма

свойства block-size и inline-size позволяют управлять размерами при различных writing-mode, где inline-size эквивалентен width (right, left), block-size - height (top, bottom)

### block-size

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

### inline-size

задает высоту или ширину блока в зависимости от написания

Если ширина не задана, общая ширина равна доступному месту в родителе при схлопывании – суммируется margin, берется максимальный.

## -moz-float-edge (-)

```scss
 {
  // учет рамок при вычислении высоты/ширины
  -moz-float-edge: content-box;
  -moz-float-edge: margin-box;
}
```

<!-- Отступы и границы ----------------------------------------------------------------------------------------------------------------------->

# Отступы и границы:

## padding

отступ от контента до рамки, при заливке заливается и padding и контент

приставки block и inline Добавляют возможность контролировать направление текста

## margin

внешние отступы, они могут быть автоматически добавлены к абзацам например, если задать margin === 0 то они схлопнуться. Схлопывание внешних отступов происходит в трёх случаях:

- Соседние элементы (siblings)
- Родительский и первый/последний дочерние элементы
- Пустые блоки
- Внешние отступы плавающих и абсолютно позиционируемых элементов никогда не схлопываются.
- margin'ы внутри флекс-контейнера не схлопываются

Если margin и padding заданы в процентах, то размеры будут взяты от inline-размера элемента

приставки block и inline Добавляют возможность контролировать направление текста

```scss
 {
  margin: auto; // Прием позволяет отдать под отступ все доступное пространство
}
```

### margin-trim (ios)

позволяет обрезать margin

## border

определит стиль для всех четырех границ сокращенная запись [border-width](#border-width) + [border-style](#border-style) + [border-color](#border-color)

приставки block и inline Добавляют возможность контролировать направление текста

```scss
.border {
  border: 4mm ridge rgba(211, 220, 50, 0.6);
}
```

### border-trbl,

border-top border-left, border-right, border-top

сокращенная запись для определения стиля, ширины и стиля границы

```scss
.border-bottom {
  border-bottom: 4mm ridge rgba(211, 220, 50, 0.6);
}
```

### border-block и border-inline

border-block и border-inline свойства, которые полагаются на направление текста:

- Свойства для верхних и нижних границ: border-block-end-color, border-block-start-color, border-block-end-style, border-block-end-width, border-block-start-width
- Свойства для левой и правой: border-inline-end-color, border-inline-start-color, border-block-start-style, border-inline-end-style, border-inline-start-style, border-inline-end-width, border-inline-start-width
- Закругления: border-start-start-radius, border-start-end-radius, border-end-start-radius, border-end-end-radius

### border-style

border-bottom-style, border-left-style, border-right-style, border-top-style

предопределенные стили для border

```scss
.border-bottom-style {
  border-bottom-style: none;
  border-bottom-style: hidden; // скрыть
  border-bottom-style: dotted; // в точку
  border-bottom-style: dashed; // в черточку
  border-bottom-style: solid; // сплошной
  border-bottom-style: double; // двойной
  border-bottom-style: groove; // двойной
  border-bottom-style: ridge; // светлый
  border-bottom-style: inset; // без заливки
  border-bottom-style: outset; // с заливкой
  // коротка запись t+r+b+l
  border-style: dashed groove none dotted;
}
```

<!--border-width  -------------------------------------------------------------------------------------------------------------------------->

### border-width

так же можно определить отдельно border-bottom-width, border-left-width, border-right-width, border-top-width

```scss
.border-width {
  // текстовые обозначения
  border-bottom-width: thin;
  border-bottom-width: medium;
  border-bottom-width: thick;

  // в абсолютных значения
  border-bottom-width: 10em;
  border-bottom-width: 3vmax;
  border-bottom-width: 6px;
  //сокращенная запись

  border-width: 0 4px 8px 12px;
}
```

### border-color

так же можно определить отдельно border-left-color-border-right-color-border-top-color

```scss
.border-color {
  border-left-color: red;
  border-left-color: #ffbb00;
  border-left-color: rgb(255 0 0);
  border-left-color: hsl(100deg 50% 25% / 75%);
  border-left-color: currentcolor;
  border-left-color: transparent;
  //короткая запись
  border-color: red yellow green transparent;
}
```

### border-image

Короткая запись для border свойств

[border-image-outset](#border-image-outset)+ [border-image-repeat](#border-image-repeat) + [border-image-slice](#border-image-slice) + [border-image-source](#border-image-source) + [border-image-width](#border-width)

```scss
.border-image {
  border-image: repeating-linear-gradient(30deg, #4d9f0c, #9198e5, #4d9f0c 20px)
    60; //
  border-image: url("/images/border.png") 27 23 / 50px 30px / 1rem round space;
}
```

#### border-image-outset

отступ

```scss
{
  // от всех границ
  border-image-outset: 1red
  // top | right | bottom | left
  border-image-outset: 7px 12px 14px 5px;
}
```

#### border-image-repeat

Позволяет растянуть картинку границы

```scss
 {
  border-image-repeat: stretch; //растяжение изображения
  border-image-repeat: repeat; //повтор
  border-image-repeat: round; //повтор
  border-image-repeat: space; //повтор
  // для нескольких границ
  border-image-repeat: round stretch;
}
```

#### border-image-slice

позволяет нарезать на количество кусков картинку и заполнить рамки

```scss
 {
  border-image-slice: 30; //позволяет распределить изображение
  border-image-slice: 30 fill; //fill - заполнит внутреннюю область
}
```

#### border-image-source

источник изображения

```scss
 {
  border-image-source: url("/media/examples/border-stars.png"); //внутренние ресурсы
  border-image-source: repeating-linear-gradient(
    45deg,
    transparent,
    #4d9f0c 20px
  ); //градиент
  border-image-source: none;
}
```

#### border-image-width

```scss
 {
  border-image-width: 30px; // в пикселях
  border-image-width: 15px 40px; //для нескольких границ
  border-image-width: 20% 8%; //в процентном соотношении
}
```

### -webkit-border-before

Рамка на верху элемента, сигнатура такая же как и у обычного border

webkit-border-before = -webkit-border-before-color + -webkit-border-before-style + -webkit-border-before-width

```scss
.webkit-border-before {
  -webkit-border-before: 5px dashed blue;
}
```

### border-radius

так же border-bottom-left-radius, border-bottom-right-radius, border-top-left-radius, border-top-right-radius, border-top-right-radius

```scss
 {
  border-bottom-right-radius: 3px;

  border-bottom-right-radius: 20%; //закругление на 1/5 часть края
  border-bottom-right-radius: 20% 10%; //20% от горизонтали и 10% от вертикали
  border-bottom-right-radius: 0.5em 1em;

  // сокращенная запись
  border-radius: 10px;
  /* top-left-and-bottom-right | top-right-and-bottom-left */
  border-radius: 10px 5%;
  /* top-left | top-right-and-bottom-left | bottom-right */
  border-radius: 2px 4px 2px;
  /* top-left | top-right | bottom-right | bottom-left */
  border-radius: 1px 0 3px 4px;
  /* The syntax of the second radius allows one to four values */
  /* (first radius values) / radius */
  border-radius: 10px / 20px;
  /* (first radius values) / top-left-and-bottom-right | top-right-and-bottom-left */
  border-radius: 10px 5% / 20px 30px;
  /* (first radius values) / top-left | top-right-and-bottom-left | bottom-right */
  border-radius: 10px 5px 2em / 20px 25px 30%;
  /* (first radius values) / top-left | top-right | bottom-right | bottom-left */
  border-radius: 10px 5% / 20px 25em 30px 35em;
}
```

# Декорирование блока

## box-shadow

Добавит тень от контейнера

- значение по горизонтали (offset-x)
- смещение по вертикали (offset-y)
- размытие тени (blur-radius)
- распространение тени (spread-radius)
- цвет тени (color)

```scss
.single-shadow {
  box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.7);
}
```

Множественное значение для теней

```scss
.multiple-shadow {
  box-shadow: 1px 1px 1px black, 2px 2px 1px black, 3px 3px 1px red, 4px 4px 1px
      red, 5px 5px 1px black, 6px 6px 1px black;
}
```

### Значение inset

inset - добавляет внутреннюю тень

```scss
button:active {
  box-shadow: inset 2px 2px 1px black, inset 2px 3px 5px rgba(0, 0, 0, 0.3),
    inset -2px -3px 5px rgba(255, 255, 255, 0.5);
}
```

## mask-border:

краткая запись следующих свойств позволяет создать маску для границ:

- - mask-border-mode: luminance | alpha использование яркости или альфа-значения в качестве маски
- - mask-border-outset: 7px 12px 14px 5px; отступы
- - mask-border-repeat: stretch | repeat | round | space применение
- - mask-border-slice: 7 12 14 5
- - mask-border-source: url(image.jpg); источник
- - mask-border-width: 5% 2em 10% auto; размеры

### -webkit-mask-box-image

webkit-mask-box-image = -webkit-mask-box-image-source + -webkit-mask-box-image-outset + -webkit-mask-box-image-repeat

альтернатива для mask-border

## box-decoration-break (-safari)

определяет поведение декорирования рамок, при переносе

```scss
 {
  // при переносе рамка будет разрываться на все строки
  -webkit-box-decoration-break: slice;
  box-decoration-break: slice;
  // при переносе рамка будет оборачивать контент каждой строки
  -webkit-box-decoration-break: clone;
  box-decoration-break: clone;
}
```

## outline - обводка

стилизация внешней рамки, которая может наезжать на соседние элементы и не влияет на определение блочной модели

Свойство обводки контента outline = outline-color + outline-style + outline-width + outline-offset

```scss
 {
  outline: 8px ridge rgba(170, 50, 220, 0.6);
}
```

### outline-color

```scss
 {
  outline-color: red;
}
```

### outline-style

стиль внешней обводки

```scss
 {
  outline-style: auto;
  outline-style: none;
  outline-style: dotted;
  outline-style: dashed;
  outline-style: solid;
  outline-style: double;
  outline-style: groove;
  outline-style: ridge;
  outline-style: inset;
  outline-style: outset;
}
```

### outline-width

ширина внешней обводки

```scss
 {
  // предопределенные стили
  outline-width: thin;
  outline-width: medium;
  outline-width: thick;
  // пользовательские
  outline-width: 1px;
  outline-width: 0.1em;
}
```

### outline-offset

отступ от обводки внешней границы

```scss
 {
  outline-offset: 4px;
  outline-offset: 0.6rem;
}
```

### -webkit-box-reflect

Позволяет добавить отражение элемента

```scss
.webkit-box-reflect {
  // где отразить
  -webkit-box-reflect: above;
  -webkit-box-reflect: below;
  -webkit-box-reflect: left;
  -webkit-box-reflect: right;

  // расстояние
  -webkit-box-reflect: below 10px;

  // маска
  -webkit-box-reflect: below 0 linear-gradient(transparent, white);
}
```

<!-- Вытекание за контейнер, скрытие и наложение --------------------------------------------------------------------------------------------->

# Вытекание за контейнер, скрытие и наложение

Возникает, при том условии, когда размер одного или группы элементов в сумме больше размера контейнера. одно из свойств блочной модели регулируется с помощью свойства overflow:

## overflow

overflow-block, overflow-inline - Для rtl

```scss
.overflow {
  // При превышении размера контента используется свойство overflow
  overflow: visible; // не воспрепятствует налеганию текста друг на друга
  overflow: scroll; //добавляет полосы прокрутки
  overflow: auto; //полосы прокрутки появляются при необходимости
  overflow: hidden; //скрывает любое содержимое выходящее за рамки
  overflow-y: scroll; // скролл по вертикали
  overflow-x: scroll; // скролл по горизонтали
}
```

- - свойство в зависимости от направленности письма:
- - - overflow-block
- - - overflow-inline
- overflow-x горизонтальный
- overflow-y вертикальный скролл

### overflow-clip-margin (-)

насколько далеко за пределами своих границ может быть нарисован элемент с перед тем, как будет обрезан

### -webkit-line-clamp (safari)

сколько строк будет обрезано

```scss
.webkit-line-clamp {
  -webkit-line-clamp: 3;
  -webkit-line-clamp: 10;
}
```

## visibility

visible | hidden | collapse не выкидывает элемент из дерева элементов, не меняет разметку

## z-index

number - позволяет выдвинуть элемент из контекста для позиционированного элемента (отличного от static) отрицательные значения понижают приоритет

Порядок наложения без z-index:

- Фон и границы корневого элемента.
- Дочерние блоки в нормальном потоке в порядке размещения(в HTML порядке).
- Дочерние позиционированные элементы, в порядке размещения (в HTML порядке).

float:

- Фон и границы корневого элемента
- Дочерние не позиционированные элементы в порядке появления в HTML
- Плавающие элементы
- Элементы, позиционируемые потомками, в порядке появления в HTML

<!-- ориентация письма ----------------------------------------------------------------------------------------------------------------------->

# BPs

## BP. Центрирование с помощью блочной модели (margin)

```css
div {
  width: 200px;
  margin: 0 auto;
}
```

width: auto – отдаст под контент, ту часть, которая останется от padding и margin
width: 100% - займет весь контейнер, но если сумма содержимого и рамок больше размера контейнера, то содержимое выпадет за переделы контейнера

## BP. скрыть элемент оставив его доступным

```scss
 {
  visibility: hidden;
  width: 0px;
  height: 0px;
}
```
