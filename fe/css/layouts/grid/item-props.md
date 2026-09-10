# Свойства grid-элемента

## grid-column / grid-row / grid-area

Управляют позицией элемента через указание линий сетки.

```
grid-column = grid-column-start / grid-column-end
grid-row    = grid-row-start / grid-row-end
grid-area   = grid-row-start / grid-column-start / grid-row-end / grid-column-end
```

> Порядок в `grid-area`: **row-start / column-start / row-end / column-end**. Это отличается от интуиции x/y — сначала вертикаль, потом горизонталь.

```scss
.item {
  // полная запись через отдельные свойства
  grid-column-start: 1;
  grid-column-end: 4;
  grid-row-start: 1;
  grid-row-end: 3;

  // сокращённая: start / end
  grid-column: 1 / 4; // от линии 1 до линии 4 = колонки 1, 2, 3
  grid-row: 1 / 3;    // от линии 1 до линии 3 = ряды 1, 2

  // самая краткая: row-start / col-start / row-end / col-end
  grid-area: 1 / 1 / 3 / 4;
}
```

### Отрицательные значения

Линии считаются с конца явной сетки (-1 = последняя линия явных треков):

```scss
.item {
  grid-column: 1 / -1; // растянуть на все явные колонки
  grid-row: 1 / -1;    // растянуть на все явные ряды
}
```

> `1 / -1` охватывает только **явную** сетку. Неявные треки (из `grid-auto-*`) не включаются.

### span

Растянуть элемент на N треков вместо указания конечной линии:

```scss
.item {
  grid-column: 2 / span 2; // от колонки 2, занять 2 колонки (до линии 4)
  grid-row: 1 / span 3;    // от ряда 1, занять 3 ряда
  grid-column: span 3;     // span без start — авто-размещение + занять 3 колонки
}
```

### Привязка к именованным областям

```scss
// grid-template-areas задаётся в контейнере
.header  { grid-area: header; }
.content { grid-area: main; }
.sidebar { grid-area: sidebar; }
.footer  { grid-area: footer; }
```

### Привязка к именованным линиям

```scss
.item {
  // использование имён линий вместо чисел
  grid-column: col-start / col-end;
  grid-row: main-start / main-end;
  // если линии с суффиксами -start/-end, можно использовать имя области
  grid-area: main; // то же что main-start / main-start / main-end / main-end
}
```

## Авто-размещение

Если `grid-column`/`grid-row` не заданы — браузер размещает элемент автоматически по правилам `grid-auto-flow`. Можно зафиксировать одну ось и оставить другую авто:

```scss
.item {
  grid-column: 1; // явная колонка, ряд — авто
  grid-row: span 2; // занять 2 ряда, но ряд — авто
}
```

## Перекрытие элементов (z-index)

Несколько элементов могут занимать одни и те же ячейки — управление слоями через `z-index`.

> Grid-элемент с `z-index !== auto` создаёт stacking context (аналогично `position: relative`).

```scss
.wrapper {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-auto-rows: 100px;
}

.box1 {
  grid-column: 1 / 4; // на всю ширину (3 колонки)
  grid-row: 1 / 3;    // на два ряда
  z-index: 2;         // будет выше box2
}

.box2 {
  grid-column-start: 1;
  grid-row: 2 / 4;    // перекрывается с box1 во втором ряду
  z-index: 1;         // будет ниже box1
}
```

## Выравнивание отдельного элемента

Переопределяют `align-items`/`justify-items` контейнера для конкретного элемента.

### align-self

Выравнивание по **block**-оси (вертикаль). Значение по умолчанию: `stretch`.

```scss
.item {
  align-self: stretch;    // (по умолчанию) растянуть на высоту трека
  align-self: start;      // к началу трека
  align-self: end;        // к концу трека
  align-self: center;     // по центру трека
  align-self: baseline;   // по базовой линии текста
  align-self: first baseline;
  align-self: last baseline;
}
```

### justify-self

Выравнивание по **inline**-оси (горизонталь). Значение по умолчанию: `stretch`. Не работает во flex.

```scss
.item {
  justify-self: stretch; // (по умолчанию)
  justify-self: start;
  justify-self: end;
  justify-self: center;
  justify-self: left;
  justify-self: right;
  // safe/unsafe — поведение при overflow
  justify-self: safe center;   // при overflow смещается к start
  justify-self: unsafe center; // при overflow может выйти за пределы
}
```

### place-self

Shorthand: `align-self justify-self`. Если одно значение — применяется к обеим осям.

```scss
.item {
  place-self: center;       // оба по центру
  place-self: start end;    // align-self: start, justify-self: end
}
```

### margin: auto

В grid `margin: auto` поглощает всё свободное место в ячейке — удобная альтернатива выравниванию:

```scss
.item {
  margin-inline-start: auto; // прижать к правому краю ячейки (nav-item в конце)
  margin: auto;              // отцентрировать в ячейке по обеим осям
}
```
