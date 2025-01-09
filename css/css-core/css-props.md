<!-- align-content (flex) --------------------------------------------------------------------------------------------------------------------->

# align-content (flex)

```scss
.align-content {
  align-content: flex-start;
  align-content: flex-end;
  align-content: center;
  align-content: space-between;
  align-content: space-around;
  align-content: stretch;
}
```

<!-- align-items (flex) ------------------------------------------------------------------------------------------------------------------>

# align-items (flex, grid)

Выравнивание по поперечной оси

```scss
.flex {
  /* высота по умолчанию  */
  align-items: stretch;
  /*по верхнему краю (к верху) */
  align-items: start;
  /*по нижнему краю (к низу) */
  align-items: end;
  /*отцентрирует */
  align-items: center;
  /* выравнивает текст внутри элементов */
  align-items: baseline;
}
```

<!-- align-self ---------------------------------------------------------------------------------------------------------------------------->

# align-self

Выравнивание элемента управляемое самим элементом

```scss
 {
  align-self: center; /* Put the item around the center */
  align-self: start; /* Put the item at the start */
  align-self: end; /* Put the item at the end */
  align-self: self-start; /* Align the item flush at the start */
  align-self: self-end; /* Align the item flush at the end */
  align-self: flex-start; /* Put the flex item at the start */
  align-self: flex-end; /* Put the flex item at the end */

  /* Baseline alignment */
  align-self: baseline;
  align-self: first baseline;
  align-self: last baseline;
  align-self: stretch; /* Stretch 'auto'-sized items to fit the container */
}
```

<!-- animation-range (scroll-driven-animation)---------------------------------------------------------------------------------------------------------------------------->

# animation-range = animation-range-start + animation-range-end

Позволяет определить настройки срабатывания анимации, относительно начала и конце шкалы

```scss
 {
  /* single keyword or length percentage value */
  animation-range: normal; /* Equivalent to normal normal */
  animation-range: 20%; /* Equivalent to 20% normal */
  animation-range: 100px; /* Equivalent to 100px normal */

  /* single named timeline range value */
  animation-range: cover; /* Представляет полный диапазон именованной временной шкалы 0% - начал входить*/
  animation-range: contain; /* элемент полностью входит*/
  animation-range: cover 20%; /* Equivalent to cover 20% cover 100% */
  animation-range: contain 100px; /* Equivalent to contain 100px cover 100% */

  /* two values for range start and end */
  animation-range: normal 25%;
  animation-range: 25% normal;
  animation-range: 25% 50%;
  animation-range: entry exit; /* exit - начал выходить */
  animation-range: cover cover 200px; /* Equivalent to cover 0% cover 200px */
  animation-range: entry 10% exit; /* entry - начал входить */
  animation-range: 10% exit 90%;
  animation-range: entry 10% 90%;
  // entry-crossing - пересек
  // exit-crossing вышел
}
```

<!-- animation-timeline (scroll-driven-animation)-------------------------------------------------------------------------------------------------------------------->

# animation-timeline (scroll-driven-animation)

Следующие типы временных шкал могут быть установлены с помощью animation-timeline:

- ременная шкала документа по умолчанию, со старта открытия страницы
- Временная шкала прогресса прокрутки, в свою очередь они делятся на:
- - Именованная временная шкала прогресса прокрутки заданная с помощью [scroll-timeline](#scroll-timeline--scroll-timeline-name-)
- - анонимная задается с помощью функции scroll()
- Временная шкала прогресса просмотра (видимость элемента) делится на
- - Именованная временная шкала прогресса [view-timeline](#view-timeline)
- - Анонимная временная шкала прогресса просмотра

```scss
.animation-timeline {
  animation-timeline: none;
  animation-timeline: auto;

  /* Single animation named timeline */
  animation-timeline: --timeline_name;

  /* Single animation anonymous scroll progress timeline */
  animation-timeline: scroll();
  animation-timeline: scroll(scroller axis);

  /* Single animation anonymous view progress timeline */
  animation-timeline: view();
  animation-timeline: view(axis inset);

  /* Multiple animations */
  animation-timeline: --progressBarTimeline, --carouselTimeline;
  animation-timeline: none, --slidingTimeline;
}
```

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

<!-- block-size ---------------------------------------------------------------------------------------------------------------------------->

# block-size

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

<!-- caret-color
 ------------------------------------------------------------------------------------------------------------------------------------------->

# caret-color

```scss
 {
  caret-color: red; //определенный цвет
  caret-color: auto; //обычно current-color
  caret-color: transparent; //невидимая
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

<!-- color-scheme ---------------------------------------------------------------------------------------------------------------------------->

# color-scheme

Применит стили темы пользователя для элементов

```scss
.color-scheme {
  color-scheme: normal;
  color-scheme: light; //Означает, что элемент может быть отображён в светлой цветовой схеме операционной системы.
  color-scheme: dark; //Означает, что элемент может быть отображён в тёмной цветовой схеме операционной системы.
  color-scheme: light dark;
}

:root {
  color-scheme: light dark;
}
```

<!-- column --------------------------------------------------------------------------------------------------------------------->

# column

Свойство позволяет разделить на столбцы текст в контейнере

```scss
 {
  // разделение текста на две колонки (*)
  -moz-column-count: 2;
  column-count: 2;
  // размер промежутка между колонками (*)
  column-gap: 4rem;
  -moz-column-gap: 4rem;
  // разделитель (*)
  column-rule: 1px solid $color-grey-light-2;
  -moz-column-rule: 1px solid $color-grey-light-2;
  //позволяет растянуть элемент по ширине всех колонок
  column-span: all;
}
```

<!-- column-fill --------------------------------------------------------------------------------------------------------------------------->

# column-fill

```scss
 {
  column-fill: auto; //Высота столбцов не контролируется.
  column-fill: balance; //Разделяет содержимое на равные по высоте столбцы.
}
```

<!-- column-gap (flex, grid, multi-column)---------------------------------------------------------------------------------------------------------------->

# column-gap (flex, grid, multi-column)

расстояние по вертикали

```scss
.column-gap {
  column-gap: auto; //1em
  column-gap: 20px;
}
```

<!-- column-rule  ------------------------------------------------------------------------------------------------------------------------->

# column-rule (multi-column)

Устанавливает цвет границы между колонками = column-rule-width + column-rule-style + column-rule-color

```scss
.column-count {
  // column-count: 3;
  column-rule: solid 8px;
  column-rule: solid blue;
  column-rule: thick inset blue;
}
```

<!-- column-rule-color --------------------------------------------------------------------------------------------------------------------->

# column-rule-color

цвет колонок

```scss
.column-rule-color {
  column-rule-color: red;
  column-rule-color: rgb(192, 56, 78);
  column-rule-color: transparent;
  column-rule-color: hsla(0, 100%, 50%, 0.6);
}
```

<!-- column-rule-style ---------------------------------------------------------------------------------------------------------------------------->

# column-rule-style

Стиль разделителя

```scss
 {
  column-rule-style: none;
  column-rule-style: hidden;
  column-rule-style: dotted;
  column-rule-style: dashed;
  column-rule-style: solid;
  column-rule-style: double;
  column-rule-style: groove;
  column-rule-style: ridge;
  column-rule-style: inset;
  column-rule-style: outset;
}
```

<!--  ---------------------------------------------------------------------------------------------------------------------------->

# column-rule-width:

Ширина колонки

```scss
 {
  column-rule-width: thin;
  column-rule-width: medium;
  column-rule-width: thick;

  /* <length> values */
  column-rule-width: 1px;
  column-rule-width: 2.5em;
}
```

<!-- column-span ---------------------------------------------------------------------------------------------------------------------------->

# column-span (multi-column)

```scss
.column-span {
  column-span: none;
  column-span: all;
}
```

```html
<!-- контейнер для определения колонок -->
<article>
  <!-- контент для распределения на колонки -->
  <h2>Header spanning all of the columns</h2>
  <p></p>
  <p></p>
  <p></p>
  <p></p>
  <p></p>
</article>
```

```scss
article {
  columns: 3;
}

h2 {
  column-span: all;
}
```

<!-- column-width --------------------------------------------------------------------------------------------------------------------->

# column-width (multi-column)

Позволяет определить максимальную ширину колонки

```scss
.container {
  column-width: 200px;
}
```

<!-- columns ---------------------------------------------------------------------------------------------------------------------------->

# columns

Устанавливает количество колонок и их ширину

```scss
 {
  /* количество */
  columns: auto;
  columns: 2;

  /* Количество и ширина */
  columns: 2 auto;
  columns: auto 12em;
  columns: auto auto;
}
```

<!-- contain  ------------------------------------------------------------------------------------------------------------------------------->

# contain

Существует четыре типа ограничения CSS: размер, макет, стиль и краска, которые устанавливаются в контейнере

```scss
 {
  contain: none;
  contain: strict; // === contain: size layout paint style
  contain: content; // === contain: layout paint style блок независимый, невидимые не будет отрисовать
  contain: size; // размер элемента может быть вычислен изолировано, работает в паре с contain-intrinsic-size
  contain: inline-size; // строчное
  contain: layout; // Внутренняя компоновка элемента изолирована от остальной части страницы
  contain: style; //Для свойств, которые могут влиять не только на элемент и его потомков, эффекты не выходят за пределы содержащего элемента
  contain: paint; //Потомки элемента не отображаются за его пределами.
}
```

- В некоторых случаях, особенно при использовании строгого значения strict, браузер может потребовать дополнительных ресурсов для оптимизации рендеринга. Поэтому важно тестировать и измерять производительность при использовании свойства.contain применяется к самому элементу и его содержимому, но не влияет на элементы, вложенные внутри него. Если требуется оптимизировать взаимодействие внутри вложенных элементов, нужно применить свойство contain к каждому из них отдельно.
- Свойство наиболее полезно в ситуациях, когда у вас есть небольшой набор элементов, которые могут быть легко изолированы и оптимизированы.
- В случае сложных макетов с большим количеством элементов, использовать contain бывает сложно и неэффективно

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

<!-- container ----------------------------------------------------------------------------------------------------------------------------->

# container

container = container-name + container-type

```scss
.container {
  container: my-layout;
  container: my-layout / size;
}
```

<!-- container-name ------------------------------------------------------------------------------------------------------------------------->

# container-name

Определяет имя контейнера

```scss
 {
  container-name: myLayout;
  container-name: myPageLayout myComponentLibrary; //несколько имен
}
```

<!--  ------------------------------------------------------------------------------------------------------------------------------------>

# container-type

```scss
 {
  container-type: normal;
  container-type: size; //по inline и block модели
  container-type: inline-size; //по строчной
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

<!-- content-visibility -------------------------------------------------------------------------------------------------------------------->

# content-visibility

Позволяет сделать содержимое контейнера невидимым. Основное применение для создание плавных анимаций, при которых контент плавно пропадает.
В анимации нужно включить transition-behavior: content-visibility

```scss
 {
  content-visibility: visible; //обычное отображение элемента
  content-visibility: hidden; // не будет доступно для поиска, фокусировки
  content-visibility: auto; //contain: content
}
```

Второе применение экономия ресурсов при рендеринге

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

<!-- flex -------------------------------------------------------------------------------------------------------------------------------->

# flex

flex-grow: 0
flex-shrink: 1
flex-basis: auto

```scss
// шорткат для flex-grow + flex-shrink + flex-basis
 {
  flex: 1 1 200px;
}
```

<!-- flex-basis ---------------------------------------------------------------------------------------------------------------------------->

# flex-basis

Устанавливает минимальное значение размера flex- элемента, если оно не установлено блочной моделью.

Разница между flex-basis 0 и 0% в том что, во втором случает элемент ужмется до своих минимальных размеров внутреннего контента

```scss
 {
  flex-basis: auto; // значение по умолчанию
  flex-basis: fill;
  flex-basis: max-content;
  flex-basis: min-content;
  flex-basis: fit-content;
  flex-basis: content; // определяет размер на основе содержимого
  flex-basis: 0; // определяет пропорционально с другими элементами
  flex-basis: 100px; // если в px то определяет минимальный размер контейнера
}
```

# flex-direction

```scss
 {
  flex-direction: row; // справа на лево, то есть блоки будут идти справа на лево
  flex-direction: column; // сверху вниз, как div-ы
  flex-direction: row-reverse; // снизу вверх
  flex-direction: column-reverse;
  // общие значения
  flex-direction: inherit;
  flex-direction: initial;
  flex-direction: revert;
  flex-direction: revert-layer;
  flex-direction: unset;
}
```

<!-- flex-flow ----------------------------------------------------------------------------------------------------------------------------->

# flex-flow

```scss
 {
  // --------------------------------------------------------------------
  // позволяет задать в одной строчке задать flex-direction + flex-wrap
  flex-flow: row wrap;
}
```

<!-- flex-wrap ----------------------------------------------------------------------------------------------------------------------------->

# flex-wrap

```scss
 {
  flex-wrap: wrap; //
}
```

<!-- gap (flex, grid)------------------------------------------------------------------------------------------------------------------->

# gap (flex, grid)

сокращенная запись gap = row-gap + column-gap

```scss
 {
  gap: 10px 20px;
}
```

<!-- grid ---------------------------------------------------------------------------------------------------------------------------->

# grid

Является сокращением для следующих свойств (значения по умолчанию)

```scss
.grid {
  // grid ===
  grid-template-rows: none;
  grid-template-columns: none;
  grid-template-areas: none;
  grid-auto-rows: auto;
  grid-auto-columns: auto;
  grid-auto-flow: row;
  grid-column-gap: 0;
  grid-row-gap: 0;
  column-gap: normal;
  row-gap: normal;

  // варианты

  grid: none;
  grid: "a" 100px "b" 1fr;
  grid: [line-name1] "a" 100px [line-name2];
  grid: "a" 200px "b" min-content;
  grid: "a" minmax(100px, max-content) "b" 20%;
  grid: 100px / 200px;
  grid: minmax(400px, min-content) / repeat(auto-fill, 50px);

  /* <'grid-template-rows'> /
   [ auto-flow && dense? ] <'grid-auto-columns'>? values */
  grid: 200px / auto-flow;
  grid: 30% / auto-flow dense;
  grid: repeat(3, [line1 line2 line3] 200px) / auto-flow 300px;
  grid: [line1] minmax(20em, max-content) / auto-flow dense 40%;

  /* [ auto-flow && dense? ] <'grid-auto-rows'>? /
   <'grid-template-columns'> values */
  grid: auto-flow / 200px;
  grid: auto-flow dense / 30%;
  grid: auto-flow 300px / repeat(3, [line1 line2 line3] 200px);
  grid: auto-flow dense 40% / [line1] minmax(20em, max-content);
}
```

<!-- grid-auto-rows ------------------------------------------------------------------------------------------------------------------------>

# grid-auto-rows и grid-auto-columns

grid-auto-rows - для автоматического распределения высоты элемента, позволяет определить высоту элемента в неявной сетке
grid-auto-columns - длины элемента

```scss
 {
  // автоматическое распределение
  grid-auto-rows: min-content;
  grid-auto-rows: max-content;
  grid-auto-rows: auto;
  //поддерживает проценты, пиксели, функции min-max
  // для сетки с множеством колонок или строк (если перенесется более одной строки)
  // если перенос будет на три ряда первый - min-content, второй - max-content
  grid-auto-rows: min-content max-content auto;
}
```

Для автоматического определения высоты в строках неявной сетки

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  grid-auto-rows: minmax(100px, auto);
}
```

<!-- grid-auto-flow ------------------------------------------------------------------------------------------------------------------------------------>

# grid-auto-flow

Определяет размещение элементов в неявной grid сетке в колонку или в ряд

```scss
 {
  grid-auto-flow: row; //вынесет элементы в новый ряд
  grid-auto-flow: column; //вынесет элементы в новую колонку
  grid-auto-flow: dense; //автоматическое распределение
  grid-auto-flow: row dense;
  grid-auto-flow: column dense;
}
```

<!-- grid-column-gap ------------------------------------------------------------------------------------------------------------------------------------>

# grid-column-gap

```scss
.grid-column-gap {
  grid-column-gap: 10px;
}
```

<!-- grid-gap ------------------------------------------------------------------------------------------------------------------------------------>

# grid-gap

```scss
.grid-gap {
  rid-gap: 10px 12px;
}
```

<!-- grid-row-gap ------------------------------------------------------------------------------------------------------------------------------------>

# grid-row-gap

```scss
.grid-row-gap {
  grid-row-gap: 10px;
}
```

<!--grid-template = grid-template-columns + grid-template-rows ------------------------------------------------------------------------------------------------------------------------------------>

# grid-template = grid-template-columns + grid-template-rows

Позволяет сформировать макет с помощью контейнера

```scss
 {
  grid-template-columns: 100px 1fr;
  grid-template-columns: [line-name] 100px;
  grid-template-columns: [line-name1] 100px [line-name2 line-name3];
  grid-template-columns: minmax(100px, 1fr);
  grid-template-columns: fit-content(40%);
  grid-template-columns: repeat(3, 200px);
  grid-template-columns: subgrid;
  grid-template-columns: masonry;

  /* <auto-track-list> values */
  grid-template-columns: 200px repeat(auto-fill, 100px) 300px;
  grid-template-columns:
    minmax(100px, max-content)
    repeat(auto-fill, 200px) 20%;
  grid-template-columns:
    [line-name1] 100px [line-name2]
    repeat(auto-fit, [line-name3 line-name4] 300px)
    100px;
  grid-template-columns:
    [line-name1 line-name2] 100px
    repeat(auto-fit, [line-name1] 300px) [line-name3];
}
```

<!-- inset- ------------------------------------------------------------------------------------------------------------------------------------>

# inset-block (якоря)

```scss
 {
  inset-block-start: 3px | 1rem | anchor(end) | calc(
      anchor(--myAnchor 50%) + 5px
    )
    | 10%
    // расположит элемент якоря, аналогично inset-block-end ,inset-inline-start, inset-inline-end
;
  inset-block: 10px 20px; // определяет начальные и конечные смещения логического блока элемента, аналогично inset-inline
  inset: ; // inset-block-start + inset-block-end + inset-inline-start + inset-inline-end
}
```

<!-- isolation ----------------------------------------------------------------------------------------------------------------------------->

# isolation

Управление контекстом стекирования

```scss
 {
  isolation: auto;
  isolation: isolate;
}
```

<!-- justify-content (flex) --------------------------------------------------------------------------------------------------------------------->

# justify-content (flex, grid)

```scss
.flex {
  justify-content: center; // Выравнивание элементов по центру
  justify-content: start; // Выравнивание элементов в начале в отличие от flex-start отчет идет от направления письма
  justify-content: end; // Выравнивание элементов в конце
  justify-content: flex-start; // Выравнивание флекс-элементов с начала
  justify-content: flex-end; // Выравнивание флекс-элементов с конца
  justify-content: left; // Выравнивание элементов по левому краю
  justify-content: right; // Выравнивание элементов по правому краю

  // Выравнивание относительно осевой линии
  justify-content: baseline;
  justify-content: first baseline;
  justify-content: last baseline;

  // Распределённое выравнивание
  justify-content: space-between; // Равномерно распределяет все элементы по ширине flex-блока. Первый элемент вначале, последний в конце
  justify-content: space-around; // Равномерно распределяет все элементы по ширине flex-блока. Все элементы имеют полноразмерное пространство с обоих концов
  justify-content: space-evenly; // Равномерно распределяет все элементы по ширине flex-блока. Все элементы имеют равное пространство вокруг
  justify-content: stretch; // Равномерно распределяет все элементы по ширине flex-блока. Все элементы имеют "авто-размер", чтобы соответствовать контейнеру
  // Выравнивание при переполнении
  justify-content: safe center;
  justify-content: unsafe center;
}
```

<!-- justify-item (grid)-------------------------------------------------------------------------------------------------------------------------->

# justify-item (grid)

игнорируется в таблицах, flex и grid сетках

```scss
 {
  justify-items: center; // Выровнять элементы по центру
  justify-items: start; // Выровнять элементы в начале
  justify-items: end; // Выровнять элементы в конце
  justify-items: flex-start; // Эквивалентно 'start'. Обратите внимание, что justify-items игнорируется в разметке Flexbox.
  justify-items: flex-end; // Эквивалентно 'end'. Обратите внимание, что justify-items игнорируется в разметке Flexbox.
  justify-items: self-start;
  justify-items: self-end;
  justify-items: left; // Выровнять элементы по левому краю
  justify-items: right; // Выровнять элементы по правому краю
  /* Исходное выравнивание */
  justify-items: baseline;
  justify-items: first baseline;
  justify-items: last baseline;
  /* Выравнивание при переполнении (только для выравнивания положения) */
  justify-items: safe center;
  justify-items: unsafe center;
  /* Унаследованное выравнивание */
  justify-items: legacy right;
  justify-items: legacy left;
  justify-items: legacy center;
}
```

<!-- justify-self (grid) ---------------------------------------------------------------------------------------------------------------------------->

# justify-self (grid)

выравнивание элемент вдоль главной оси. не работает в flex и табличных контейнерах

```scss
 {
  // Positional alignment
  justify-self: center; // Pack item around the center
  justify-self: start; // Pack item from the start
  justify-self: end; // Pack item from the end
  justify-self: flex-start; // Equivalent to 'start'. Note that justify-self is ignored in flexbox layouts.
  justify-self: flex-end; // Equivalent to 'end'. Note that justify-self is ignored in flexbox layouts.
  justify-self: self-start;
  justify-self: self-end;
  justify-self: left; // Pack item from the left
  justify-self: right; // Pack item from the right
  justify-self: anchor-center;

  // Baseline alignment
  justify-self: baseline;
  justify-self: first baseline;
  justify-self: last baseline;

  // Overflow alignment (for positional alignment only)
  justify-self: safe center;
  justify-self: unsafe center;
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

<!-- mix-blend-mode  ---------------------------------------------------------------------------------------------------------------------->

# mix-blend-mode

правило смешивания наслаивающих изображений и фонов https://developer.mozilla.org/en-US/docs/Web/CSS/blend-mode

```scss
 {
  mix-blend-mode: lighten | overlay;
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

<!-- offset ---------------------------------------------------------------------------------------------------------------------------->

# offset

```scss
 {
  offset: 10px 30px;

  /* Offset path */
  offset: ray(45deg closest-side);
  offset: path("M 100 100 L 300 100 L 200 300 z");
  offset: url(arc.svg);

  /* Offset path with distance and/or rotation */
  offset: url(circle.svg) 100px;
  offset: url(circle.svg) 40%;
  offset: url(circle.svg) 30deg;
  offset: url(circle.svg) 50px 20deg;

  /* Including offset anchor */
  offset: ray(45deg closest-side) / 40px 20px;
  offset: url(arc.svg) 2cm / 0.5cm 3cm;
  offset: url(arc.svg) 30deg / 50px 100px;
}
```

<!-- offset-anchor ---------------------------------------------------------------------------------------------------------------------------->

# offset-anchor

Позволяет определить где будет находится элемент относительно прямой при движение по линии [offset](./css-props.md/#offset)

```scss
 {
  offset-anchor: top;
  offset-anchor: bottom;
  offset-anchor: left;
  offset-anchor: right;
  offset-anchor: center;
  offset-anchor: auto;

  /* <percentage> values */
  offset-anchor: 25% 75%;

  /* <length> values */
  offset-anchor: 0 0;
  offset-anchor: 1cm 2cm;
  offset-anchor: 10ch 8em;

  /* Edge offsets values */
  offset-anchor: bottom 10px right 20px;
  offset-anchor: right 3em bottom 10px;
}
```

<!-- offset-path --------------------------------------------------------------------------------------------------------------------------->

# offset-path

Позволяет задать путь движения

```scss
 {
  offset-path: ray(45deg closest-side contain);
  offset-path: ray(contain 150deg at center center);
  offset-path: ray(45deg);

  /* URL */
  offset-path: url(#myCircle);

  /* Basic shape */
  offset-path: circle(50% at 25% 25%);
  offset-path: ellipse(50% 50% at 25% 25%);
  offset-path: inset(50% 50% 50% 50%);
  offset-path: polygon(30% 0%, 70% 0%, 100% 50%, 30% 100%, 0% 70%, 0% 30%);
  offset-path: path(
    "M 0,200 Q 200,200 260,80 Q 290,20 400,0 Q 300,100 400,200"
  );
  offset-path: rect(5px 5px 160px 145px round 20%);
  offset-path: xywh(0 5px 100% 75% round 15% 0);

  /* Coordinate box */
  offset-path: content-box;
  offset-path: padding-box;
  offset-path: border-box;
  offset-path: fill-box;
  offset-path: stroke-box;
  offset-path: view-box;
}
```

<!-- orphans ------------------------------------------------------------------------------------------------------------------------------->

# orphans

Минимальное число строк, которое можно оставить внизу фрагмента перед разрывом фрагмента. Значение должно быть положительным.

```scss
 {
  orphans: 3;
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

<!-- overflow -------------------------------------------------------------------------------------------------------------------------------->

# overflow

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

<!-- page-break-before --------------------------------------------------------------------------------------------------------------------->

# page-break-before, page-break-after, page-break-inside

Устанавливает разрывы для печати на странице до или после элемента

```scss
 {
  page-break-before: auto;
  page-break-before: always;
  page-break-before: avoid;
  page-break-before: left;
  page-break-before: right;
  page-break-before: recto;
  page-break-before: verso;
}
```

<!-- place-items (grid fle-------------------------------------------------------------------------------------------------------------->

# place-items (grid, flex)

короткая запись place-items = align-items + justify-items

```scss
 {
  place-items: end center;
}
```

<!-- place-self (grid, flex) --------------------------------------------------------------------------------------------------------------->

# place-self (grid, flex)

place-self = align-self + justify-self

```scss
 {
  place-self: stretch center;
}
```

<!-- pointer-events ------------------------------------------------------------------------------------------------------------------------>

# pointer-events

Определяет цель для курсора

```scss
 {
  pointer-events: auto;
  pointer-events: none;
  // для svg
  pointer-events: visiblePainted;
  pointer-events: visibleFill;
  pointer-events: visibleStroke;
  pointer-events: visible;
  pointer-events: painted;
  pointer-events: fill;
  pointer-events: stroke;
  pointer-events: bounding-box;
  pointer-events: all;
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

<!-- row-gap (flex, grid)------------------------------------------------------------------------------------------------------------------->

# row-gap (flex, grid)

расстояние по горизонтали

```scss
 {
  row-gap: 20px;
}
```

<!-- resize -------------------------------------------------------------------------------------------------------------------------------->

# resize (-safari)

Позволяет растягивать элемент

```scss
 {
  resize: none; //отключает растягивание
  resize: both; //тянуть можно во все стороны
  resize: horizontal;
  resize: vertical;
  resize: block; // в зависимости от writing-mode и direction
  resize: inline; // в зависимости от writing-mode и direction
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

<!-- scroll-timeline ---------------------------------------------------------------------------------------------------------------------------->

# scroll-timeline = scroll-timeline-name +

```scss
 {
  //
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

<!-- scrollbar-color  ---------------------------------------------------------------------------------------------------------------------->

# scrollbar-color

Цвет полосы прокрутки

```scss
 {
  // первое значение - полоса прокрутки, второе - ползунок
  scrollbar-color: rebeccapurple green;
}
```

<!--scroll-timeline------------------------------------------------------------------------------------------------------------------------->

# scroll-timeline

```scss
 {
  //scroll-timeline-name  scroll-timeline-axis
  scroll-timeline: --custom_name_for_timeline block;
  scroll-timeline: --custom_name_for_timeline inline;
  scroll-timeline: --custom_name_for_timeline y;
  scroll-timeline: --custom_name_for_timeline x;
  scroll-timeline: none block;
  scroll-timeline: none inline;
  scroll-timeline: none y;
  scroll-timeline: none x;
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

<!-- view-timeline ------------------------------------------------------------------------------------------------------------------------->

# view-timeline = view-timeline-name + view-timeline-axis

Определяет временную шкалу для анимации от видимости элемента

```scss
 {
  view-timeline: --custom_name_for_timeline block;
  view-timeline: --custom_name_for_timeline inline;
  view-timeline: --custom_name_for_timeline y;
  view-timeline: --custom_name_for_timeline x;
  view-timeline: none block;
  view-timeline: none inline;
  view-timeline: none y;
  view-timeline: none x;

  //view-timeline-name значения
  view-timeline-name: none;
  view-timeline-name: --custom_name_for_timeline;

  //view-timeline-axis значения
  view-timeline-axis: block;
  view-timeline-axis: inline;
  view-timeline-axis: y;
  view-timeline-axis: x;
}
```

<!-- view-timeline-inset ------------------------------------------------------------------------------------------------------------------->

# view-timeline-inset

Корректирует срабатывание анимации относительно скролла

Если значение положительное, положение начала/конца анимации будет перемещено внутри области прокрутки на указанную длину или процент.
Если значение отрицательное, то позиция начала/конца анимации будет перемещена за пределы области прокрутки на указанную длину или процент, т. е. анимация начнется до того, как появится в области прокрутки, или закончится после того, как анимация покинет область прокрутки.

```scss
.view-timeline-inset {
  //* Single value */
  view-timeline-inset: auto;
  view-timeline-inset: 200px;
  view-timeline-inset: 20%;

  /* Two values */
  view-timeline-inset: 20% auto;
  view-timeline-inset: auto 200px;
  view-timeline-inset: 20% 200px;
}
```

<!-- word-wrap --------------------------------------------------------------------------------------------------------------------------->

# word-break

```scss
 {
  word-wrap: "normal" | "break-word" | "inherit"; //перенос строки при переполнении
}
```
