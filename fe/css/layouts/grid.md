Позволяет управлять потоком в двух направлениях в пространстве (в горизонтальном и вертикальном). Создание grid макета с помощью display: grid или display: inline-grid. При активации grid на контейнере, создаются линии - границы grid-сетки. Grid- ячейка это часть сетки разделенная линиями, grid-область - это объединенные части сетки.Grid поддерживает направление письма

- при указании высоты height для контейнера с элементами расположенными друг за другом - высота будет делится на кол-во контейнеров

<!-- свойства grid контейнера ---------------------------------------------------------------------------------------------------------------->

# свойства grid контейнера

## grid

Является сокращением для следующих свойств (значения по умолчанию)

```scss
.grid {
  // grid, значения по умолчанию про display:grid
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

  grid: 200px / auto-flow;
  grid: 30% / auto-flow dense;
  grid: repeat(3, [line1 line2 line3] 200px) / auto-flow 300px;
  grid: [line1] minmax(20em, max-content) / auto-flow dense 40%;

  grid: auto-flow / 200px;
  grid: auto-flow dense / 30%;
  grid: auto-flow 300px / repeat(3, [line1 line2 line3] 200px);
  grid: auto-flow dense 40% / [line1] minmax(20em, max-content);
}
```

## grid-template:

grid-template-areas + grid-template-rows + grid-template-columns

Основное свойство для контейнера, которой может задать сетку для всех элементов внутри этой стеки

```scss
.container {
  grid-template: "header   header   header   header" 50px // значения высоты grid-template-rows
    "article  article   .       sidebar" auto // значения высоты grid-template-rows
    "footer   footer   footer   footer" 50px // значения высоты grid-template-rows
    /1fr 1fr 50px 1fr; //значения колонок grid-template-columns
}
.container > * {
}
.header {
  grid-area: header;
}
.article {
  grid-area: article;
}
.aside {
  grid-area: aside;
}
.footer {
  grid-area: footer;
}
```

### grid-template-rows и grid-template-columns

grid-template-rows - определяет имена и размеры рядов
grid-template-columns - определяет имена и размер колонок

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

```scss
.container {
  grid-template: repeat(7, 5vw) / repeat(8, 1fr);
}
```

### именованные колонки и ряды

Позволяет добавить названия для линий в сетке чтобы избавиться от числовых значений в пользу именованных
При добавлении к названиям линий -end, -start позволяет располагать элементы с помощью одного слова

```scss
.wrapper {
  display: grid;
  grid-template-columns: [main-start] 1fr [content-start] 1fr [content-end] 1fr [main-end];
  grid-template-rows: [main-start] 100px [content-start] 100px [content-end] 100px [main-end];
}

// использование
.box1 {
  grid-column-start: main-start;
  grid-row-start: main-start;
  grid-row-end: main-end;
}

.thing {
  grid-area: content;
}
```

именованные линии сетки можно использовать с repeat

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(12, [col-start] 1fr);
  //с двумя значениями
  grid-template-columns: repeat(4, [col1-start] 1fr [col2-start] 3fr);
}
// обращение
.item1 {
  grid-column: col-start / col-start 5;
  // со span
  grid-column: col-start 7 / span 3;
}
```

## явная и неявная сетки

Существует явная и неявная сетка. Неявная сетка образуется при размещении вне контента. При количестве элементов превышающим количество в grid сетке, используются свойства определяющие неявную сетку:

### grid-auto-flow

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

### grid-auto-rows и grid-auto-columns

grid-auto-rows - для автоматического распределения высоты элемента, позволяет определить высоту элемента в неявной сетке
grid-auto-columns - длины элемента

```scss
 {
  // автоматическое распределение
  grid-auto-rows: min-content;
  grid-auto-rows: max-content;
  grid-auto-rows: auto;
  grid-auto-rows: 100px;
  //поддерживает проценты, пиксели, функции min-max
  // для сетки с множеством колонок или строк (если перенесется более одной строки)
  // если перенос будет на три ряда первый - min-content, второй - max-content
  grid-auto-rows: min-content max-content auto;
  // если перенос будет на три ряда первый - 100px, второй - 200px
  grid-auto-rows: 100px 200px;
}
```

все возможные значения

```scss
.grid-auto-columns {
  grid-auto-columns/*rows*/: 100px;
  grid-auto-columns/*rows*/: 20cm;
  grid-auto-columns/*rows*/: 50vmax;

  /* <percentage> values */
  grid-auto-columns/*rows*/: 10%;
  grid-auto-columns/*rows*/: 33.3%;

  /* <flex> values */
  grid-auto-columns/*rows*/: 0.5fr;
  grid-auto-columns/*rows*/: 3fr;

  /* minmax() values */
  grid-auto-columns/*rows*/: minmax(100px, auto);
  grid-auto-columns/*rows*/: minmax(max-content, 2fr);
  grid-auto-columns/*rows*/: minmax(20%, 80vmax);

  /* fit-content() values === min(max-content, max(auto, argument)) */
  grid-auto-columns/*rows*/: fit-content(400px);
  grid-auto-columns/*rows*/: fit-content(5cm);
  grid-auto-columns/*rows*/: fit-content(20%);

  /* multiple track-size values */
  grid-auto-columns/*rows*/: min-content max-content auto;
  grid-auto-columns/*rows*/: 100px 150px 390px;
  grid-auto-columns/*rows*/: 10% 33.3%;
  grid-auto-columns/*rows*/: 0.5fr 3fr 1fr;
  grid-auto-columns/*rows*/: minmax(100px, auto) minmax(max-content, 2fr) minmax(20%, 80vmax);
  grid-auto-columns/*rows*/: 100px minmax(100px, auto) 10% 0.5fr fit-content(
      400px
    );

  /* Global values */
  grid-auto-columns/*rows*/: inherit;
  grid-auto-columns/*rows*/: initial;
  grid-auto-columns/*rows*/: revert;
  grid-auto-columns/*rows*/: revert-layer;
  grid-auto-columns/*rows*/: unset;
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

## grid-template-areas

именованные области. Пример простой сетки

```html
<div class="wrapper">
  <!-- отдельный ряд для header -->
  <div class="header">Header</div>
  <!-- ряд для sidebar и  content -->
  <div class="sidebar">Sidebar</div>
  <div class="content">Content</div>
  <div class="footer">Footer</div>
</div>
```

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(9, 1fr);
  grid-auto-rows: minmax(100px, auto);
  grid-template-areas:
    "hd hd hd hd   hd   hd   hd   hd   hd"
    "sd sd sd main main main main main main"
    "ft ft ft ft   ft   ft   ft   ft   ft";
}

.header {
  grid-area: hd;
}
.footer {
  grid-area: ft;
}
.content {
  grid-area: main;
}
.sidebar {
  grid-area: sd;
}
```

Для задания пустоты используется точка

```scss
.wrapper {
  // ...
  grid-template-areas:
    "hd hd hd hd   hd   hd   hd   hd   hd"
    "sd sd sd main main main main main main"
    ".  .  .  ft   ft   ft   ft   ft   ft";
}
```

Для макета, элементы которого могут принимать разную форму. sd займет место и в колонках и рядах

```scss
.wrapper {
  // ...
  grid-template-areas:
    "hd hd hd hd   hd   hd   hd   hd   hd"
    "sd sd sd main main main main main main"
    "sd sd sd  ft  ft   ft   ft   ft   ft";

  grid-template-columns: repeat(4, 1fr);
  // включение опции по инициализации каждого grid-контейнера
  // схема по занятым колонкам и рядам ... - пустое место использование
  grid-template-areas:
    "header   header   header   header"
    "article  article   ...     sidebar"
    "footer   footer   footer   footer";
}
```

## grid-gap:

промежутки между колонками и рядами

```scss
.grid-gap {
  rid-gap: 10px 12px;
}
```

### grid-row-gap

```scss
.grid-row-gap {
  grid-row-gap: 10px;
}
```

### grid-column-gap

```scss
.grid-column-gap {
  grid-column-gap: 10px;
}
```

## Функции в контейнере

### fr

1fr - единица измерения относительной величины, определенная, как относительная единица контейнера.
Если в grid-template указано относительных единиц меньше, чем элементов, то остальные элементы перенесутся на следующую строку

```scss
.container {
  display: grid;
  // 200px первая колонка, оставшиеся место поделить 1:2 выделить 1/3 - второй, 2/3 - третьей
  grid-template-columns: 200px 1fr 2fr;
  // сожмет контент до минимально возможного размера
  grid-template-rows: 1fr min-content 6rem 1fr;
  // заберет все возможное пространство grid-контейнера
  grid-template-columns: max-content;
}
```

### minmax

Позволяет выбрать максимальное, а если значение меньше минимального позволяет выбрать минимальное. С использованием auto - выбрать все доступное пространство

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  // растянуть до самого высокого
  grid-auto-rows: minmax(100px, auto);
}
```

адаптивные методы с помощью minmax

```html
<div class="container">
  <div class="item">1</div>
  <div class="item">2</div>
  <div class="item">3</div>
  <div class="item">4</div>
  <div class="item">5</div>
  <div class="item">6</div>
  <div class="item">7</div>
</div>
```

```css
.container {
  background-color: #ccc;
  padding: 1.5rem;
  display: grid;
  gap: 10px 15px;

  grid-template-rows: 100px 200px;
  /* для выпавших из строк - автоматическая высота*/
  /* приделать footer к низу */
  grid-template-rows: 100px minmax(500px, 1fr) 200px;
  grid-auto-rows: minmax(100px, auto);
  /* меняем порядок из колонок в ряды и наоборот */
  /* grid-auto-flow: column; */
}
.item {
  background-color: peru;
  border-radius: 15px;
  padding: 1rem;
}
```

### repeat

Позволяет избежать дублирования

```scss
.container {
  display: grid;
  //повторяющийся контент
  grid-template-columns: repeat(3, 1fr) //1fr 1fr 1fr;
  //5 повторяющихся треков по 1fr и 2fr - итого 10 колонок
  grid-template-columns: repeat(5, 1fr 2fr);
}
```

```html
<div class="container">
  <div class="item">1</div>
  ...
  <div class="item">9</div>
</div>
```

#### auto-fill и auto-fit

auto-fill - позволяет задать повторяемые элементы заданного значения с помощью repeat()

```html
<div class="container">
  <div class="item">1</div>
  <div class="item">2</div>
  <div class="item">3</div>
</div>
```

```scss
.container {
  //4 колонки по 100px вне зависимости от ширины view port
  grid-template-columns: repeat(4, 100px);
  //самостоятельно решит сколько колонок поместится в зависимости он ширины VP
  // есть VP = 300, и есть 3 элемента по 100px, то он их определит в 3 ряда до тех пор пока VP >= 300
  grid-template-columns: repeat(auto-fill, 100px);
  //здесь все так же, но только при VP>300 элементы будут растянуты
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  // в отличие от auto-fill будет занимать все свободное место
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  // чтобы занимать минимальное возможное
  grid-template-columns: minmax(min-content, max-content);
}
```

### fit-content

формула fit-content() values === min(max-content, max(auto, argument))

```scss
.fit-content {
  //Если изображение больше, дорожка перестает расти на 400 пикселях
  grid-auto-columns/*_rows_*/: fit-content(400px);
  grid-auto-columns/*_rows_*/: fit-content(5cm);
  grid-auto-columns/*_rows_*/: fit-content(20%);
}
```

<!-- Свойства grid-элементов ----------------------------------------------------------------------------------------------------------------->

# Свойства grid-элементов (grid-area, grid-column, grid-row, grid-row-start, grid-row-end, grid-column-start, grid-column-end)

Эта группа свойств позволяет управлять расположением элементов внутри сутки, при создании сетки непосредственно внутри каждого элемента.

- grid-column = grid-column-start + grid-column-end
- grid-row = grid-row-start + grid-row-end
- grid-area = grid-column + grid-row

Пример использования 4 колонки в два ряда

```scss
.wrapper {
  display: grid;
  gap: 10px;
}
.wrapper > div {
  background-color: gray;
  min-width: 100px;
  min-height: 100px;
  border: 1px solid #000;
}
// первый ряд 1 колонка
.item1 {
  // развернутое задание границ
  grid-column-start: 1;
  grid-column-end: 4;
  grid-row-start: 1;
  grid-row-end: 3;
  // короткое
  grid-column: 1 / 4;
  grid-row: 1 / 3;
  // самое короткое block-start / block-end / inline-start / inline-end
  grid-area: 1 / 4 / 1 / 3;
}
// первый ряд 2 колонка
.item2 {
  grid-column: 2 / 3;
}
.item3 {
  grid-column: 3 / 4;
}
.item4 {
  grid-column: 4 / 5;
}
// второй ряд
.item5 {
  // большой блок справа
  grid-column: 2 / 3;
  grid-row: 1 / 3;
}
.item6 {
  grid-column: 1 / 2;
  grid-row: 1 / 2;
}
.item7 {
  grid-column: 1 / 2;
  grid-row: 2 / 3;
}
```

```html
<div class="wrapper">
  <div class="item1"></div>
  <div class="item2"></div>
  <div class="item3"></div>
  <div class="item4"></div>
</div>
<div class="wrapper">
  <div class="item5"></div>
  <div class="item6"></div>
  <div class="item7"></div>
  <img src="./assets/css/grid-column-grid-row.png" width="300" height="300" />
</div>
```

- если задавать отрицательные значения то порядок считается с конца.

```scss
.item {
  // Это может помочь растянуть элемент по всей длине/ширине
  grid-column: 1 / -1;
}
```

### span - растягивает элемент

растягивает на определенное количество единиц. Если задать только span то расположат в начале

```scss
.item {
  grid-column: 2 / span 2;
  grid-row: 1 / span 3;
}
```

- Если задать пересекающиеся значения, то элементы сетки будут накладываться друг на друга. Управление осуществляется с помощью z-index

<!-- Выравнивание ---------------------------------------------------------------------------------------------------------------------------->

# Выравнивание

Две оси - ось блока (колонки) и ось ряда (inline)

- Выравнивание так же работает с margin: auto
- writing-mode влияет на то как отображается сетка

## Выравнивание по block оси:

### align-items

Выравнивание по block оси происходит так же как и во flex сетках

- [align-items - выравнивает элементы оси блока внутри ряда](./flex.md#align-items-flex-grid)

### align-content

Распределит контент в сетке по блоковой оси, если есть свободное место (все свойства с content про свободно место)

[align-content](./flex.md#align-content-flex)

Так же работают методы с margin:auto

```scss
.container {
  display: grid;
  gap: 10px 20px;

  //для случая использования фракций вертикаль/горизонталь
  justify-items: center;
  align-items: center;
  // justify-items + align-items = place-items
  place-items: end start;

  // если используем пиксели а не фракции
  // justify-content: space-evenly;
  // align-content: center;
  // place-content = justify-content + align-content
  place-content: center space-evenly;
}
.aside {
  // для отдельного элемента
  place-self: start end;
  background-color: maroon;
}
```

### align-self

Выравнивание элемента управляемое самим элементом

```scss
.align-self {
  align-self: center; // Put the item around the center
  align-self: start; // Put the item at the start
  align-self: end; // Put the item at the end
  align-self: self-start; // Align the item flush at the start
  align-self: self-end; // Align the item flush at the end
  align-self: flex-start; // Put the flex item at the start
  align-self: flex-end; // Put the flex item at the end

  // Baseline alignment
  align-self: baseline;
  align-self: first baseline;
  align-self: last baseline;
  align-self: stretch; // Stretch 'auto'-sized items to fit the container
}
```

## Выравнивание по inline оси:

### justify-self

выравнивание элемент вдоль главной оси. не работает в flex и табличных контейнерах

```scss
.justify-self {
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

Если сетка использует область, которая меньше чем контейнер

### justify-content

[выравнивание элемент вдоль главной оси, внутри grid контейнера для оси inline если есть свободное место](./flex.md#justify-content-flex-grid)

### justify-items

определяет атрибут по умолчанию justify-self дял каждого grid элемента

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

<!-- использование z-index ------------------------------------------------------------------------------------------------------------------->

# использование z-index

z-index может расположить две ячейки грида одну поверх другой

```html
<div class="wrapper">
  <div class="box box1">One</div>
  <div class="box box2">Two</div>
  <div class="box box3">Three</div>
  <div class="box box4">Four</div>
  <div class="box box5">Five</div>
</div>
```

```scss
.wrapper {
  display: grid;
  // три колонки
  grid-template-columns: repeat(3, 1fr);
  grid-auto-rows: 100px;
}
.box1 {
  // колонка на всю ширину 3fr
  grid-column-start: 1;
  grid-column-end: 4;
  //начало - самый верх
  grid-row-start: 1;
  grid-row-end: 3;
  // будет выше
  z-index: 2;
}
.box2 {
  // тоже с самого верха
  grid-column-start: 1;
  grid-row-start: 2;
  grid-row-end: 4;
  // будет перекрыт
  z-index: 1;
}
```

# sub-grid

- в sub-grid отсутствует неявная сетка
- gap можно переопределить
- наименование grid линий:
- - может быть аналогичным
- - или template-columns: subgrid [line1] [line2] [line3] [line4]

```scss
// пример использования
.grid {
  display: grid;
  grid-template-columns: repeat(9, 1fr);
  grid-template-rows: repeat(4, minmax(100px, auto));
}

.item {
  display: grid;
  grid-column: 2 / 7;
  grid-row: 2 / 4;
  // вложенный элемент будет ориентироваться по трекам внешнего грида
  grid-template-columns: subgrid;
  grid-template-rows: subgrid;
}
```

# masonry

кладка - будет выравниваться по самому большому элементу

```scss
.grid {
  display: grid;
  // выровняет все ряды в одну высоту
  grid-template-rows: masonry;
  // выровняет все колонки в одну ширину
  grid-template-columns: masonry;
}
```

<!-- BPs ------------------------------------------------------------------------------------------------------------------------------------->

# BPs

## BP.Абсолютное позиционирование в grid-сетке

grid сетку можно использовать для позиционирования задав контейнеру position: relative

## BP. адаптивная сетка

```html
<div class="wrapper">
  <header>
    <h1>Header</h1>
  </header>
  <article>
    <h2>Title</h2>
    <h2>Lorem</h2>
  </article>
  <aside>
    <h3>Aside</h3>
    <blockquote>Nice quite</blockquote>
  </aside>
</div>
```

```scss
.wrapper {
  display: grid;
  gap: 10px;
}
.wrapper > * {
}
.header {
}
.article {
}
.aside {
}

/* два разных варианта для VP */
@media (min-width: 767px) {
  .header {
    grid-column: 1 / 3;
    grid-row: 1 / 2;
  }
  .article {
    grid-column: 1 / 2;
    grid-row: 2 / 3;
  }
  .aside {
    grid-column: 2 / 3;
    grid-row: 2 / 3;
  }
}

@media (min-width: 1024px) {
  .article {
    grid-column: 2 / 3;
    grid-row: 2 / 3;
  }
  .aside {
    grid-column: 1 / 2;
    grid-row: 2 / 3;
  }
}
```

## BP. центрирование с помощью grid

```scss
.container {
  display: grid;
  place-items: center;
}
```

## BP. липкий footer

```scss
.wrapper {
  min-height: 100%;
  display: grid;
  grid-template-rows: auto 1fr auto;
}
```

## BP. grid размещение текста поверх картинки

```html
<div class="container">
  <div class="image"></div>
  <div class="text">xxx</div>
</div>
```

```css
.container {
  display: grid;
  justify-items: center;
}
/* два элемента наезжают друг на друга */
.image {
  grid-column: 1 / -1;
  grid-row: 1 / -1;
  width: 100px;
  height: 100px;
}

.text {
  grid-column: 1 / -1;
  grid-row: 1 / -1;
  /* центрирование текста */
  align-self: center;
}
```

## BP. Карточка

```html
<div class="media">
  <div class="img">
    <img
      src="https://mdn.github.io/shared-assets/images/examples/balloons_square.jpg"
      alt="Balloons"
    />
  </div>

  <div class="content">
    <p>
      Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis vehicula
      vitae ligula sit amet maximus. Nunc auctor neque ipsum, ac porttitor elit
      lobortis ac. Vivamus ultrices sodales tellus et aliquam. Pellentesque
      porta sit amet nulla vitae luctus. Praesent quis risus id dolor venenatis
      condimentum.
    </p>
  </div>
  <div class="footer">An optional footer goes here.</div>
</div>

<div class="media">
  <div class="img">
    <img
      src="https://mdn.github.io/shared-assets/images/examples/sharp-account_box-24px.svg"
      width="80px"
      alt="Account"
    />
  </div>
  <div class="content">
    <p>
      Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis vehicula
      vitae ligula sit amet maximus. Nunc auctor neque ipsum, ac porttitor elit
      lobortis ac. Vivamus ultrices sodales tellus et aliquam. Pellentesque
      porta sit amet nulla vitae luctus. Praesent quis risus id dolor venenatis
      condimentum.
    </p>
  </div>
  <div class="footer"></div>
</div>
```

```scss
body {
  font: 1.2em sans-serif;
}

img {
  max-width: 100%;
}

p {
  margin: 0 0 1em 0;
}

@media (min-width: 500px) {
  .media {
    display: grid;
    //Если изображение больше, дорожка перестает расти на 200 пикселях
    grid-template-columns: fit-content(200px) 1fr;
    grid-template-rows: 1fr auto;
    grid-template-areas:
      "image content"
      "image footer";
    grid-gap: 20px;
    margin-bottom: 4em;
  }

  .media-flip {
    grid-template-columns: 1fr fit-content(250px);
    grid-template-areas:
      "content image"
      "footer image";
  }

  .img {
    grid-area: image;
  }

  .content {
    grid-area: content;
  }

  .footer {
    grid-area: footer;
  }
}
```

## BP. grid сетки

### 3 колонки

```html
<div class="wrapper">
  <header class="main-head">The header</header>
  <nav class="main-nav">
    <ul>
      <li><a href="">Nav 1</a></li>
      <li><a href="">Nav 2</a></li>
      <li><a href="">Nav 3</a></li>
    </ul>
  </nav>
  <article class="content">
    <h1>Main article area</h1>
    <p>
      In this layout, we display the areas in source order for any screen less
      that 500 pixels wide. We go to a two column layout, and then to a three
      column layout by redefining the grid, and the placement of items on the
      grid.
    </p>
  </article>
  <aside class="side">Sidebar</aside>
  <div class="ad">Advertising</div>
  <footer class="main-footer">The footer</footer>
</div>
```

```scss
.main-head {
  grid-area: header;
}
.content {
  grid-area: content;
}
.main-nav {
  grid-area: nav;
}
.side {
  grid-area: sidebar;
}
.ad {
  grid-area: ad;
}
.main-footer {
  grid-area: footer;
}

// для мобильной версии
.wrapper {
  display: grid;
  grid-gap: 20px;
  grid-template-areas:
    "header"
    "nav"
    "content"
    "sidebar"
    "ad"
    "footer";
}

@media (min-width: 500px) {
  .wrapper {
    grid-template-columns: 1fr 3fr;
    grid-template-areas:
      "header  header"
      "nav     nav"
      "sidebar content"
      "ad      footer";
  }
  nav ul {
    display: flex;
    justify-content: space-between;
  }
}

@media (min-width: 700px) {
  .wrapper {
    grid-template-columns: 1fr 4fr 1fr;
    grid-template-areas:
      "header header  header"
      "nav    content sidebar"
      "nav    content ad"
      "footer footer  footer";
  }
  nav ul {
    flex-direction: column;
  }
}
```

### 12 колонок

```html
<div class="wrapper">
  <div class="item1">Start column line 1, span 3 column tracks.</div>
  <div class="item2">
    Start column line 6, span 4 column tracks. 2 row tracks.
  </div>
  <div class="item3">Start row 2 column line 2, span 2 column tracks.</div>
  <div class="item4">
    Start at column line 3, span to the end of the grid (-1).
  </div>
</div>
```

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(12, [col-start] 1fr);
  grid-gap: 20px;
}

.item1 {
  grid-column: col-start / span 3;
}
.item2 {
  grid-column: col-start 6 / span 4;
  grid-row: 1 / 3;
}
.item3 {
  grid-column: col-start 2 / span 2;
  grid-row: 2;
}
.item4 {
  grid-column: col-start 3 / -1;
  grid-row: 3;
}
```

или

```html
<div class="wrapper">
  <header class="main-head">The header</header>
  <nav class="main-nav">
    <ul>
      <li><a href="">Nav 1</a></li>
      <li><a href="">Nav 2</a></li>
      <li><a href="">Nav 3</a></li>
    </ul>
  </nav>
  <article class="content">
    <h1>Main article area</h1>
    <p>
      In this layout, we display the areas in source order for any screen less
      that 500 pixels wide. We go to a two column layout, and then to a three
      column layout by redefining the grid, and the placement of items on the
      grid.
    </p>
  </article>
  <aside class="side">Sidebar</aside>
  <div class="ad">Advertising</div>
  <footer class="main-footer">The footer</footer>
</div>
```

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(12, [col-start] 1fr);
  grid-gap: 20px;
}

.wrapper > * {
  grid-column: col-start / span 12;
}

@media (min-width: 500px) {
  .side {
    grid-column: col-start / span 3;
    grid-row: 3;
  }
  .ad {
    grid-column: col-start / span 3;
    grid-row: 4;
  }
  .content,
  .main-footer {
    grid-column: col-start 4 / span 9;
  }
  nav ul {
    display: flex;
    justify-content: space-between;
  }
}

@media (min-width: 700px) {
  .main-nav {
    grid-column: col-start / span 2;
    grid-row: 2 / 4;
  }
  .content {
    grid-column: col-start 3 / span 8;
    grid-row: 2 / 4;
  }
  .side {
    grid-column: col-start 11 / span 2;
    grid-row: 2;
  }
  .ad {
    grid-column: col-start 11 / span 2;
    grid-row: 3;
  }
  .main-footer {
    grid-column: col-start / span 12;
  }
  nav ul {
    flex-direction: column;
  }
}
```

## BP. список

```html
<ul class="listing">
  <li>
    <h2>Item One</h2>
    <div class="body"><p>The content of this listing item goes here.</p></div>
    <div class="cta"><a href="">Call to action!</a></div>
  </li>
  <li>
    <h2>Item Two</h2>
    <div class="body"><p>The content of this listing item goes here.</p></div>
    <div class="cta"><a href="">Call to action!</a></div>
  </li>
  <li class="wide">
    <h2>Item Three</h2>
    <div class="body">
      <p>The content of this listing item goes here.</p>
      <p>This one has more text than the other items.</p>
      <p>Quite a lot more</p>
      <p>Perhaps we could do something different with it?</p>
    </div>
    <div class="cta"><a href="">Call to action!</a></div>
  </li>
  <li>
    <h2>Item Four</h2>
    <div class="body"><p>The content of this listing item goes here.</p></div>
    <div class="cta"><a href="">Call to action!</a></div>
  </li>
  <li>
    <h2>Item Five</h2>
    <div class="body"><p>The content of this listing item goes here.</p></div>
    <div class="cta"><a href="">Call to action!</a></div>
  </li>
</ul>
```

```scss
.listing {
  list-style: none;
  margin: 2em;
  display: grid;
  grid-gap: 20px;
  // автоматическое распределение, не меньше 200зч
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
}
// внутренности карточки
.listing li {
  border: 1px solid #ffe066;
  border-radius: 5px;
  display: flex;
  flex-direction: column;
}
.listing .cta {
  margin-top: auto;
  border-top: 1px solid #ffe066;
  padding: 10px;
  text-align: center;
}
.listing .body {
  padding: 10px;
}
//
.listing {
  list-style: none;
  margin: 2em;
  display: grid;
  grid-gap: 20px;
  // автоматически определит перенос
  grid-auto-flow: dense;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
}
.listing .wide {
  grid-column-end: span 2;
}
```

## Равная ширина для всех элементов

- для flex так не получится так как он распределяет свободное место равномерно

```scss
.grid-container {
  grid: none / auto-flow minmax(min-content, 1fr);
}
```
