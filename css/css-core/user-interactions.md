# Поля ввода

## accent-color

определяет цвета интерфейсов взаимодействия с пользователем

для input type: checkbox, radio, range, progress

```scss
.accent-color {
  accent-color: red;
}
```

- - А именно: input type="checkbox", input type="radio", input type="range", progress

## appearance

Определяет внешний вид для элементов взаимодействия

```scss
.appearance {
  appearance: none; //выключает стилизацию
  appearance: auto; //значение предопределенные ОС
  appearance: menulist-button; //auto
  appearance: textfield; //auto
  appearance: button;
  appearance: checkbox;
}
```

## caret-color

определяет свойство указателя

```scss
 {
  caret-color: red; //определенный цвет
  caret-color: auto; //обычно current-color
  caret-color: transparent; //невидимая
}
```

## -moz-orient

Определяет расположение элемента горизонтально, вертикально

```scss
.moz-orient {
  -moz-orient: inline | block | horizontal | vertical;
}
```

## -webkit-text-security

символ на который будет заменен текст

```scss
.webkit-text-security {
  -webkit-text-security: circle | disc | square | none;
}
```

## -moz-user-input (-)

Запрет на ввод в поле ввода

```scss
.moz-user-input {
  -moz-user-input: auto;
  -moz-user-input: none;
}
```

<!-- Курсор ---------------------------------------------------------------------------------------------------------------------------------->

# Курсор

## cursor

определение вида курсора, при наводки на элемент

```scss
// типы стандартных курсоров
cursor: auto;
cursor: pointer;
cursor: zoom-out;
cursor: context-menu;
cursor: help;
cursor: pointer;
cursor: progress;
cursor: wait;
cursor: cell;
cursor: crosshair;
cursor: text;
cursor: vertical-text;
cursor: alias;
cursor: copy;
cursor: move;
cursor: no-drop;
cursor: not-allowed;
cursor: all-scroll;
cursor: col-resize;
cursor: row-resize;
cursor: n-resize;
cursor: e-resize;
cursor: s-resize;
cursor: w-resize;
cursor: ne-resize;
cursor: nw-resize;
cursor: se-resize;
cursor: sw-resize;
cursor: ew-resize;
cursor: ns-resize;
cursor: nesw-resize;
cursor: nwse-resize;
cursor: zoom-in;
cursor: zoom-out;
cursor: grab;
cursor: grabbing;

// использование изображения в качестве курсора + fallback
cursor: url(hand.cur), pointer;

// использование изображения в качестве курсора + координаты + fallback
cursor: url(cursor_1.png) 4 12, auto;
cursor: url(cursor_2.png) 2 2, pointer;

// использование изображения в качестве курсора + координаты + fallback в виде других изображений
cursor: url(cursor_1.svg) 4 5, url(cursor_2.svg), /* …, */ url(cursor_n.cur) 5 5,
  progress;
```

## pointer-events

Определяет цель для курсора
позволяет указать, что может быть целью курсора

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

# resize (-safari)

позволяет сделать элемент растягиваемым

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

# user-select

метод выделения текста курсором

[user-select](./text.md#user-select)

# скролл

## Внешний вид полоски скролла

Поведение при скролле:

### scrollbar-width (-safari)

auto | thin | none;

### scrollbar-color (-safari)

Цвет полосы прокрутки

```scss
 {
  // первое значение - полоса прокрутки, второе - ползунок
  scrollbar-color: rebeccapurple green;
}
```

### scrollbar-gutter

```scss
.scrollbar-gutter {
  scrollbar-gutter: auto;
  scrollbar-gutter: stable; // отступ есть по обе стороны
  scrollbar-gutter: both-edges;
}
```

## Поведение прокрутки

### scroll-behavior

auto | smooth для поведения прокрутки

auto | stable | oth-edges stable -

## scroll-snap-type

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

## scroll-snap-align

center | start | end позволяет при прокрутки фиксировать позицию элемента

## scroll-margin

px позволяет прокрутить в определенное место элемента с определенным margin, является сокращенной записью для scroll-margin-right + scroll-margin-bottom + scroll-margin-left, при нуле поместит элемент в середину

### scroll-margin-inline:

scroll-margin-inline-start + scroll-margin-inline-end

###scroll-margin-block

scroll-margin-block-start + scroll-margin-block-end

### scroll-padding:

px позволяет прокрутить в определенное место при scroll-snap,
scroll-padding = scroll-padding-bottom + scroll-padding-left + scroll-padding-top + scroll-padding-right

### scroll-padding-inline

scroll-padding-inline-start + scroll-padding-inline-end

### scroll-padding-block

scroll-padding-block-start + scroll-padding-block-end

### scroll-snap-stop

normal | always придает дискретность к прокрутке

### overscroll-behavior

auto | contain | none - поведение при достижении конца скролла шорткат для:

- - overscroll-behavior-x
- - overscroll-behavior-y
- - overscroll-behavior-block, overscroll-behavior-inline для поведения с учетом направленности текста

```html
<article class="scroller">
  <section>
    <h2>Section one</h2>
  </section>
  <section>
    <h2>Section two</h2>
  </section>
  <section>
    <h2>Section three</h2>
  </section>
</article>
```

```scss
.scroller {
  height: 300px;
  overflow-y: scroll;
  scroll-snap-type: y mandatory;
}

.scroller section {
  scroll-snap-align: start;
}
```

### overflow-anchor (-safari)

auto | none определяет поведения прокрутки, при добавлении элементов

Сделать скролл дискретным (при прокрутке привязывался к позиции)

## ::-webkit-scrollbar - псевдоэлементы группы scrollbar:

```scss
 {
  ::-webkit-scrollbar-button {
  }
  ::-webkit-scrollbar:horizontal {
  }
  ::-webkit-scrollbar-thumb {
  }
  ::-webkit-scrollbar-track {
  }
  ::-webkit-scrollbar-track-piece {
  }
  ::-webkit-scrollbar:vertical {
  }
  ::-webkit-scrollbar-corner {
  }
  ::-webkit-resizer {
  }
}
```

# -moz-user-focus (-)

Отключение возможности фокусировки на элементе

```scss
.moz-user-focus {
  -moz-user-focus: ignore | normal | none;
}
```
