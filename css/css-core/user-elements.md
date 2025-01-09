<!-- Элементы взаимодействия с пользователем ----------------------------------------------------------------------------------------------->

# accent-color

определяет цвета интерфейсов взаимодействия с пользователем

```scss
.accent-color {
  accent-color: red;
}
```

- - А именно: input type="checkbox", input type="radio", input type="range", progress

# appearance

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

- [caret-color определяет свойство указателя](./css-props.md/#caret-color)
- [cursor определение вида курсора, при наводки на элемент](./css-props.md/#cursor)
- [pointer-events позволяет указать, что может быть целью курсора](./css-props.md/#pointer-events)
- [resize позволяет сделать элемент растягиваемым](./css-props.md/#resize)

# скролл

Поведение при скролле:

- scroll-behavior: auto | smooth для поведения прокрутки
- (нет в Safari) scrollbar-width auto | thin | none;
- (нет в Safari)[scrollbar-color цвет scrollbar ](./css-props.md/#scrollbar-color)
- (нет в Safari) scrollbar-gutter: auto | stable | oth-edges - ;
- (нет в Safari) overflow-anchor: auto | none определяет поведения прокрутки, при добавлении элементов

Сделать скролл дискретным (при прокрутке привязывался к позиции)

- [scroll-snap-type как строго привязывается прокрутка ](./css-props.md/#scroll-snap-type)
- scroll-snap-align: center | start | end позволяет при прокрутки фиксировать позицию элемента
- scroll-margin: px позволяет прокрутить в определенное место элемента с определенным margin, является сокращенной записью для scroll-margin-right + scroll-margin-bottom + scroll-margin-left, при нуле поместит элемент в середину
- scroll-margin-inline = scroll-margin-inline-start + scroll-margin-inline-end
- scroll-margin-block = scroll-margin-block-start + scroll-margin-block-end
- scroll-padding: px позволяет прокрутить в определенное место при scroll-snap, коротка запись для группы scroll-padding-bottom + scroll-padding-left + scroll-padding-top + scroll-padding-right
- scroll-padding-inline = scroll-padding-inline-start + scroll-padding-inline-end
- scroll-padding-block = scroll-padding-block-start + scroll-padding-block-end

свойства scroll-padding- и scroll-margin- могут помочь в ситуации когда заголовок остается в фиксированном месте

- scroll-snap-stop: normal | always придает дискретность к прокрутке
- overscroll-behavior: auto | contain | none - поведение при достижении конца скролла шорткат для:
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

::-webkit-scrollbar - псевдоэлементы группы scrollbar:

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
