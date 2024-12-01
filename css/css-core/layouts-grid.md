# Grid

Позволяет управлять потоком в двух направлениях в пространстве (в горизонтальном и вертикальном). Создание grid макета с помощью display: grid или display: inline-grid. При активации grid на контейнере, создаются линии - границы grid-сетки. Grid- ячейка это часть сетки разделенная линиями, grid-область - это объединенные части сетки.Grid поддерживает направление письма

[grid - является краткой записью для нескольких свойств](./css-props.md/#grid)

# grid-column-start, grid-column-end, grid-row-start, grid-row-end, grid-column, grid-row, grid-area (grid-element)

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

## span - растягивает элемент

растягивает на определенное количество единиц. Если задать только span то расположат в начале

```scss
.item {
  grid-column: 2 / span 2;
  grid-row: 1 / span 3;
}
```

- Если задать пересекающиеся значения, то элементы сетки будут накладываться друг на друга. Управление осуществляется с помощью z-index

# grid-template === grid-template-column + grid-template-crow (grid-container)

Основное свойство для контейнера, которой может задать сетку для всех элементов внутри этой стеки

- [grid-template - Свойство позволяет определить макет в двух направлениях](./css-props.md/#grid-template--grid-template-columns--grid-template-rows)

```scss
.container {
  grid-template: repeat(7, 5vw) / repeat(8, 1fr);
}
```

## fr

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

## repeat

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

## auto-fill и auto-fit

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

## именованные колонки и ряды

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

# явная и неявная сетки

Существует явная и неявная сетка. Неявная сетка образуется при размещении вне контента. При количестве элементов превышающим количество в grid сетке, используются свойства определяющие неявную сетку:

- [grid-auto-flow - определяет размещение элементов в неявной grid сетке grid-auto-flow](./css-props.md/#grid-auto-flow)
- [grid-auto-rows и grid-auto-columns - определяет размер неявной сетки](./css-props.md/#grid-auto-rows-и-grid-auto-columns)

# minmax

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

# grid-gap - промежутки между колонками и рядами

- [короткая запись grid-gap = grid-row-gap + grid-column-gap](./css-props.md/#grid-gap)
- - [grid-row-gap](./css-props.md/#grid-row-gap)
- - [grid-column-gap](./css-props.md/#grid-column-gap)

# areas - области

Пример простой сетки

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

# grid-template = grid-template-areas + grid-template-rows + grid-template-columns

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

# Выравнивание alignment

Две оси - ось блока (колонки) и ось ряда (inline)

Выравнивание на оси блока:

- [align-self - выравнивание в контейнере](./css-props.md/#align-self)
- [align-items - выравнивает элементы оси блока внутри элемента](./css-props.md/#align-items-flex)

Выравнивание на оси ряда:

- [justify-items - выравнивает элементы вдоль главной оси, внутри своего контейнера](./css-props.md#justify-self-grid)
- [justify-self - индивидуально расположение элемента](./css-props.md#justify-self-grid)

Если сетка использует область, которая меньше чем контейнер

- [justify-content - выравнивание элемент вдоль главной оси, внутри grid контейнера для оси inline](./css-props.md#justify-content-flex)
- align-content - для оси блока

Так же работаю методы с margin:auto

```css
.container {
  display: grid;
  gap: 10px 20px;

  /* -- для случая использования фракций вертикаль/горизонталь */
  justify-items: center;
  align-items: center;
  /* justify-items + align-items = place-items */
  place-items: end start;

  /* если используем пиксели а не фракции*/
  /* justify-content: space-evenly;*/
  /* align-content: center; */
  /* justify-content + align-content = place-content */
  place-content: center space-evenly;
}
.aside {
  /* для отдельного элемента*/
  place-self: start end;
  background-color: maroon;
}
```

# Sub-grid

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
  grid-template-columns: subgrid;
  grid-template-rows: subgrid;
}
```

# Абсолютное позиционирование в grid-сетке

grid сетку можно использовать для позиционирования задав контейнеру position: relative

# BP. адаптивная сетка

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

# BP. центрирование с помощью grid

```scss
.container {
  display: grid;
  place-items: center;
}
```

# BP. липкий footer

```scss
.wrapper {
  min-height: 100%;
  display: grid;
  grid-template-rows: auto 1fr auto;
}
```

# BP. grid размещение текста поверх картинки

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

# BP. Карточка

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
