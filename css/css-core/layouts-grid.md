<!-- Сетки. Grid -------------------------------------------------------------------------------------------------------------------------->

# Сетки. Grid

Позволяет управлять потоком в двух направлениях в пространстве (в горизонтальном и вертикальном). Создание grid макета с помощью display: grid или display: inline-grid. При активации grid на контейнере, создаются линии - границы grid-сетки. Grid- ячейка это часть сетки разделенная линиями, grid-область - это объединенные части сетки

Grid поддерживает направление письма

[Свойство grid является краткой записью для нескольких свойств](./css-props.md/#grid)

## grid-colum, grid-row, grid-area создание сетки с помощью элементов

При создании сетки непосредственно внутри каждого элемента. Пример использования 4 колонки в два ряда

grid-area = grid-row-start + grid-column-start + grid-row-end + grid-column-end

```scss
// заголовок:
.wrapper {
  // grid on
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

## grid-template === grid-column + grid-row

```scss
.container {
  grid-template: repeat(7, 5vw) / repeat(8, 1fr);
}
```

- Если задать пересекающиеся значения, то элементы сетки будут накладываться друг на друга. Управление осуществляется с помощью z-index
- если задавать отрицательные значения то порядок считается с конца. Это может помочь растянуть элемент по всей длине/ширине

```scss
.item {
  grid-column: 1 / -1;
}
```

## span

растягивает на определенное количество единиц. Если задать только span то расположат в начале

```scss
.item {
  grid-column: 2 / span 2;
  grid-row: 1 / span 3;
}
```

## паттерны и фракции grid-template-column grid-template-row

Существует явная и неявная сетка. Неявная сетка образуется при размещении вне контента

- [Свойство позволяет определить макет в двух направлениях](./css-props.md/#grid-template--grid-template-columns--grid-template-rows)
  Свойства определяющие неявную сетку:
- [определяет размещение элементов в неявной grid сетке grid-auto-flow](./css-props.md/#grid-auto-flow)
- [grid-auto-rows grid-auto-columns определяет размер неявной сетки](./css-props.md/#grid-auto-rows-и-grid-auto-columns)

## fr фракции

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
  grid-template-columns: repeat(3, 1fr);
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

## minmax

Позволяет выбрать максимальное, а если значение меньше минимального позволяет выбрать минимальное. С использованием auto - выбрать все доступное пространство

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  // растянуть до самого высокого
  grid-auto-rows: minmax(100px, auto);
}
```

## именованные колонки и ряды

Позволяет добавить названия для линий в сетке чтобы избавиться от числовых значений

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
```

При добавлении к названиям линий -end, -start позволяет располагать элементы с помощью одного слова

```scss
.wrapper {
  display: grid;
  grid-template-columns: [main-start] 1fr [content-start] 1fr [content-end] 1fr [main-end];
  grid-template-rows: [main-start] 100px [content-start] 100px [content-end] 100px [main-end];
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

## Промежутки между колонками и рядами grid-gap

- [короткая запись grid-gap = grid-row-gap + grid-column-gap](./css-props.md/#grid-gap)
- - [grid-row-gap](./css-props.md/#grid-row-gap)
- - [grid-column-gap](./css-props.md/#grid-column-gap)

## areas

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
}
```

```scss
body {
  color: white;
}
.container {
  display: grid;
  gap: 10px 20px;
  grid-template-columns: repeat(4, 1fr);
  // включение опции по инициализации каждого grid-контейнера
  // схема по занятым колонкам и рядам ... - пустое место  использование
    grid-template-areas:
        'header     header      header      header'
        'article    article     ...         sidebar'
        'footer     footer      footer      footer';   
  // grid-template: areas | grid-template-rows | grid-template-columns;
  grid-template:
    "header     header      header      header"  50px // значения высоты grid-template-rows
    "article    article     .           sidebar" auto // значения высоты grid-template-rows
    "footer     footer      footer      footer"  50px // значения высоты grid-template-rows
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
  grid-area: aside
  }
.footer {
  grid-area: footer
  }

```

## Выравнивание alignment

Две оси - ось блока (колонки) и ось ряда (inline)

ВЫравнивание на оси блока:

- [align-self - выравнивание в элементе](./css-props.md/#align-self)
- [align-items - выравнивает элементы оси блока](./css-props.md/#align-items-flex)

Выравнивание на оси ряда:

- [justify-content - выравнивание элемент вдоль главной оси, внутри grid контейнера](./css-props.md#justify-content-flex)
- [justify-items - выравнивает элементы вдоль главной оси, внутри своего контейнера](./css-props.md#justify-self-grid)
- [justify-self - индивидуально расположение элемента](./css-props.md#justify-self-grid)

```css
.container{
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
  /*  для отдельного элемента*/
  place-self: start end;
  background-color: maroon;
}

```

## адаптивные методы

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

### auto-fill и auto-fit

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

## Sub-grid (Firefox 71)

```scss
.box1 {
  grid-column-start: 1;
  grid-column-end: 4;
  grid-row-start: 1;
  grid-row-end: 3;
  display: grid;
  grid-template-columns: subgrid; //вложенная сетка будет использовать родительскую сетку
}
```

## Абсолютное позиционирование в grid-сетке

grid сетку можно использовать для позиционирования задав контейнеру position: relative

### BP. адаптивная сетка

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
