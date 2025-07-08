# Внешний вид полоски скролла

Поведение при скролле:

## scrollbar-width (-s)

auto | thin | none;

## scrollbar-color (-s)

Цвет полосы прокрутки

```scss
.scrollbar-color {
  // первое значение - полоса прокрутки, второе - ползунок
  scrollbar-color: rebeccapurple green;
}
```

## scrollbar-gutter

Оставлять или нет место для прокрутки если даже прокрутки нет

```scss
.scrollbar-gutter {
  scrollbar-gutter: auto;
  scrollbar-gutter: stable; // отступ есть по обе стороны, нужно для того что бы резервировать место под scroll bar при открытии модальных окон
  scrollbar-gutter: both-edges;
}
```

## ::-webkit-scrollbar - псевдоэлементы группы scrollbar:

```scss
.webkit-scrollbar {
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

# Поведение прокрутки

## scroll-behavior

auto | smooth для поведения прокрутки

## scroll-snap-type

определяет строгость привязки применяется е контейнеру

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
  scroll-snap-type: y proximity; // привязка может произойти , но не обязательно, если точка прокрутки близка к границе
}
```

## scroll-snap-align

применяется к дочерним элементам

center | start | end позволяет при прокрутки фиксировать позицию элемента

## scroll-margin

применяется к контейнеру

px - позволяет прокрутить в определенное место элемента с определенным margin, является сокращенной записью для scroll-margin-right + scroll-margin-bottom + scroll-margin-left, при нуле поместит элемент в середину

## scroll-margin-inline:

scroll-margin-inline-start + scroll-margin-inline-end

## scroll-margin-block

scroll-margin-block-start + scroll-margin-block-end

## scroll-padding:

применяется к контейнеру

px позволяет прокрутить в определенное место при scroll-snap,
scroll-padding = scroll-padding-bottom + scroll-padding-left + scroll-padding-top + scroll-padding-right

## scroll-padding-inline

scroll-padding-inline-start + scroll-padding-inline-end

## scroll-padding-block

scroll-padding-block-start + scroll-padding-block-end

## scroll-snap-stop

normal | always придает дискретность к прокрутке

## overscroll-behavior

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

## overflow-anchor (-s)

auto | none определяет поведения прокрутки, при добавлении элементов

Сделать скролл дискретным (при прокрутке привязывался к позиции)

# -moz-user-focus (-)

Отключение возможности фокусировки на элементе

```scss
.moz-user-focus {
  -moz-user-focus: ignore | normal | none;
}
```
