# Свойства grid-контейнера

## grid

Shorthand для всех grid-свойств контейнера одновременно. На практике трудно читать — лучше задавать свойства отдельно.

```scss
.grid {
  // значения по умолчанию при display: grid
  grid-template-rows: none;    // нет явных рядов
  grid-template-columns: none; // нет явных колонок
  grid-template-areas: none;
  grid-auto-rows: auto;        // неявные ряды — по содержимому
  grid-auto-columns: auto;
  grid-auto-flow: row;         // авто-размещение идёт по рядам
  column-gap: normal;          // нет зазоров
  row-gap: normal;
}
```

Варианты shorthand-синтаксиса:

```scss
.grid {
  // <rows> / <columns>
  grid: 100px / 200px;
  grid: repeat(3, 1fr) / repeat(4, 1fr);

  // <areas> с высотами рядов / ширины колонок
  grid: "a" 100px "b" 1fr / 1fr 2fr;
  grid: [line1] "a" 100px [line2] / 1fr;

  // auto-flow по колонкам: <rows> / auto-flow <col-size>
  grid: 200px / auto-flow;
  grid: repeat(3, 200px) / auto-flow 300px;
  grid: 30% / auto-flow dense;

  // auto-flow по рядам: auto-flow <row-size> / <columns>
  grid: auto-flow / 200px;
  grid: auto-flow 300px / repeat(3, 200px);
  grid: auto-flow dense / 30%;
}
```

## grid-template

Shorthand для `grid-template-areas` + `grid-template-rows` + `grid-template-columns`:

```scss
.container {
  // синтаксис: "area" высота-ряда / ширины колонок
  grid-template:
    "header  header  header" 50px   // первый ряд — 50px
    "article article sidebar" auto  // второй ряд — по содержимому
    "footer  footer  footer" 50px   // третий ряд — 50px
    / 1fr 1fr 50px 1fr;             // 4 колонки
}

// элементы привязываются к именованным областям
.header  { grid-area: header; }
.article { grid-area: article; }
.aside   { grid-area: sidebar; }
.footer  { grid-area: footer; }
```

```scss
.container {
  // 7 рядов по 5vw, 8 равных колонок
  grid-template: repeat(7, 5vw) / repeat(8, 1fr);
}
```

## grid-template-columns / grid-template-rows

Задают явные треки сетки. `none` = нет явных треков, все элементы попадают в неявную сетку.

```scss
.container {
  // фиксированные и гибкие треки
  grid-template-columns: 100px 1fr;
  grid-template-columns: 200px 1fr 2fr; // 200px + треть + две трети от остатка

  // именованные линии (в квадратных скобках)
  grid-template-columns: [start] 100px [middle] 1fr [end];
  grid-template-columns: [line-name1] 100px [line-name2 line-name3]; // несколько имён на линию

  // функции
  grid-template-columns: minmax(100px, 1fr);  // минимум 100px, максимум — 1fr
  grid-template-columns: fit-content(40%);    // растёт по контенту, не более 40%
  grid-template-columns: repeat(3, 200px);    // три колонки по 200px

  // ключевые слова
  grid-template-columns: subgrid;           // наследовать треки от grid-родителя
  grid-template-columns: masonry;           // (-ff -sf -ch -ed) экспериментально

  // авто-размещение (repeat с auto-fill/auto-fit)
  grid-template-columns: 200px repeat(auto-fill, 100px) 300px;
  grid-template-columns:
    [col-start] 100px [col-end]
    repeat(auto-fit, [col-start col-end] 300px)
    100px;
}
```

### Именованные линии

Вместо числовых индексов — имена. Суффиксы `-start` / `-end` позволяют ссылаться на область одним словом.

```scss
.wrapper {
  display: grid;
  // линии получают имена; main-start/main-end образуют неявную область "main"
  grid-template-columns: [main-start] 1fr [content-start] 1fr [content-end] 1fr [main-end];
  grid-template-rows:    [main-start] 100px [content-start] 100px [content-end] 100px [main-end];
}

.box1 {
  grid-column-start: main-start;  // по имени линии
  grid-row-start: main-start;
  grid-row-end: main-end;
}

// краткое: element занимает область "content" (content-start → content-end)
.thing {
  grid-area: content;
}
```

Именованные линии с `repeat` — браузер создаёт нумерованные экземпляры:

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(12, [col-start] 1fr); // линии col-start 1..12
  grid-template-columns: repeat(4, [col1-start] 1fr [col2-start] 3fr); // два имени на паттерн
}

.item1 {
  grid-column: col-start / col-start 5; // от первой col-start до пятой
  grid-column: col-start 7 / span 3;   // от 7-й col-start, занять 3 трека
}
```

## Явная и неявная сетки

**Явная** — треки из `grid-template-*`. **Неявная** — треки, созданные браузером автоматически когда элементов больше чем мест.

### grid-auto-flow

Направление авто-размещения элементов в неявной сетке.

```scss
.container {
  grid-auto-flow: row;          // (по умолчанию) новые элементы идут в новый ряд
  grid-auto-flow: column;       // новые элементы идут в новую колонку
  grid-auto-flow: dense;        // заполняет дыры меньшими элементами
  grid-auto-flow: row dense;
  grid-auto-flow: column dense;
}
```

> `dense` может нарушать визуальный порядок относительно DOM. Это негативно влияет на доступность: tab-order и screen readers следуют DOM, не визуальному порядку.

### grid-auto-rows / grid-auto-columns

Размер треков неявной сетки. Принимает те же значения что `grid-template-*`, но без именованных линий.

```scss
.container {
  grid-auto-rows: auto;                // (по умолчанию) высота по содержимому
  grid-auto-rows: 100px;               // фиксированная высота
  grid-auto-rows: minmax(100px, auto); // минимум 100px, максимум — по содержимому
  grid-auto-rows: min-content;
  grid-auto-rows: max-content;

  // чередование: нечётные ряды 100px, чётные 200px
  grid-auto-rows: 100px 200px;
  // три чередующихся значения для 3+ неявных рядов
  grid-auto-rows: min-content max-content auto;
}
```

Типичное применение — неизвестное количество строк с минимальной высотой:

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  grid-auto-rows: minmax(100px, auto); // каждый неявный ряд — минимум 100px
}
```

## grid-template-areas

Именованные области — визуальное представление сетки строками.

- Каждая строка кавычек = один ряд
- Количество токенов в строке = количество колонок (должно совпадать в каждой строке)
- `.` или `...` — пустая ячейка
- Область **должна быть прямоугольной** — L-образные, T-образные формы невалидны
- Каждое имя автоматически создаёт линии `name-start` и `name-end`

```html
<div class="wrapper">
  <div class="header">Header</div>
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

.header  { grid-area: hd; }
.footer  { grid-area: ft; }
.content { grid-area: main; }
.sidebar { grid-area: sd; }
```

Пустое место и sidebar на два ряда:

```scss
.wrapper {
  grid-template-areas:
    "header  header  header  header"
    "article article .       sidebar" // . — пустая ячейка
    "footer  footer  footer  footer";
}

// sidebar на два ряда (sd занимает 2-й и 3-й ряд)
.wrapper {
  grid-template-areas:
    "hd hd hd hd   hd   hd   hd   hd   hd"
    "sd sd sd main main main main main main"
    "sd sd sd  ft   ft   ft   ft   ft   ft";
}
```

## gap

Промежутки между треками. `grid-gap`, `grid-row-gap`, `grid-column-gap` — устаревшие алиасы (deprecated).

```scss
.container {
  gap: 1rem;         // одинаковый зазор по рядам и колонкам
  gap: 10px 20px;    // row-gap column-gap
  row-gap: 10px;     // только между рядами
  column-gap: 20px;  // только между колонками
}
```

> `gap` добавляет отступ только **между** треками, не перед первым и не после последнего.

## Выравнивание (контейнер)

Две оси: **block** (вертикаль) и **inline** (горизонталь). `writing-mode` меняет их ориентацию.

### align-items

Выравнивание всех элементов по **block**-оси (вертикаль) внутри своего трека. Значение по умолчанию: `stretch`.

```scss
.container {
  align-items: stretch;    // (по умолчанию) растянуть на всю высоту трека
  align-items: start;      // к началу трека
  align-items: end;        // к концу трека
  align-items: center;     // по центру трека
  align-items: baseline;   // по базовой линии первой строки текста
}
```

### align-content

Распределяет свободное место по **block**-оси между рядами. Работает только если высота контейнера больше суммы высот всех рядов.

```scss
.container {
  align-content: start;         // ряды прижаты к началу
  align-content: end;           // ряды прижаты к концу
  align-content: center;        // ряды по центру
  align-content: stretch;       // ряды растягиваются чтобы заполнить контейнер
  align-content: space-between; // первый и последний прижаты к краям, остальные равномерно
  align-content: space-around;  // равные отступы вокруг каждого ряда
  align-content: space-evenly;  // равные отступы между всеми рядами и краями
}
```

### justify-items

Выравнивание всех элементов по **inline**-оси (горизонталь) внутри своей ячейки. Значение по умолчанию: `stretch`. Не работает во flex-контейнерах.

```scss
.container {
  justify-items: stretch; // (по умолчанию) растянуть на всю ширину ячейки
  justify-items: start;   // к началу ячейки
  justify-items: end;     // к концу ячейки
  justify-items: center;  // по центру ячейки
  justify-items: baseline;
  // legacy — для выравнивания в блочных контекстах (не grid)
  justify-items: legacy right;
  justify-items: legacy left;
  justify-items: legacy center;
}
```

### justify-content

Распределяет свободное место по **inline**-оси между колонками. Работает только если суммарная ширина колонок (фиксированных) меньше ширины контейнера.

```scss
.container {
  justify-content: start;         // колонки прижаты к началу
  justify-content: end;           // колонки прижаты к концу
  justify-content: center;        // колонки по центру
  justify-content: stretch;       // колонки растягиваются
  justify-content: space-between; // первая и последняя прижаты к краям
  justify-content: space-around;
  justify-content: space-evenly;
}
```

### place-items / place-content

Shorthands для выравнивания: первое значение — по block-оси, второе — по inline-оси. Если одно значение — применяется к обеим осям.

```scss
.container {
  // place-items: align-items justify-items
  place-items: center;        // оба по центру
  place-items: end start;     // align-items: end, justify-items: start

  // place-content: align-content justify-content
  place-content: center space-evenly;
  place-content: center; // оба по центру
}
```

Самый быстрый способ отцентрировать содержимое:

```scss
.center {
  display: grid;
  place-items: center; // или place-content: center
}
```
