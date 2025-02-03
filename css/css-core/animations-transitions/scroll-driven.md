scroll-driven animations (-ff -safari)

Существует две шкалы прогресса - прокрутка шкалы прогресса (от 0% до 100%) и временная шкала прогресса в зависимости от видимости объекта

# scroll-timeline (на контейнере) (-ff -safari)

scroll-timeline = scroll-timeline-name + scroll-timeline-axis

для определения именованной шкалы прокрутки

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

# animation-timeline (на объекте анимации)

Следующие типы временных шкал могут быть установлены с помощью animation-timeline:

- временная шкала документа по умолчанию, со старта открытия страницы
- Временная шкала прогресса прокрутки, в свою очередь они делятся на:
- - Именованная временная шкала прогресса прокрутки заданная с помощью scroll-timeline
- - анонимная задается с помощью функции scroll()
- Временная шкала прогресса просмотра (видимость элемента) делится на
- - Именованная временная шкала прогресса view-timeline
- - Анонимная временная шкала прогресса просмотра

```scss
.animation-timeline {
  animation-timeline: none;
  animation-timeline: auto;

  // ссылка на временную шкалу контейнера выше
  animation-timeline: --timeline_name;

  // ссылка на скролл контейнера
  animation-timeline: scroll();
  animation-timeline: scroll(scroller axis);

  // Single animation anonymous view progress timeline
  animation-timeline: view();
  animation-timeline: view(axis inset);

  // Multiple animations
  animation-timeline: --progressBarTimeline, --carouselTimeline;
  animation-timeline: none, --slidingTimeline;
}
```

## scroll()

Функция для отслеживания временной шкалы анонимной анимации зависящей от скролла

```scss
.scroll {
  animation-timeline: scroll();

  animation-timeline: scroll(
    nearest
  ); // Ближайший предок текущего элемента, имеющий полосы прокрутки на любой из осей. Это значение по умолчанию
  animation-timeline: scroll(root); //корневой элемент
  animation-timeline: scroll(self); //скролл самого себя

  // Values for selecting the axis
  animation-timeline: scroll(block); //перпендикулярно тексту
  animation-timeline: scroll(inline); //по направлению текста
  animation-timeline: scroll(y); //по осям
  animation-timeline: scroll(x);

  // комбинация
  animation-timeline: scroll(block nearest); // по умолчанию
  animation-timeline: scroll(inline root);
  animation-timeline: scroll(x self);
}
```

# BPs

## PB. пример

именованная анимация scroll. scroll-timeline должен быть задан на родительском контейнере, а анимация должна применяться к дочернему. Что бы избежать это поведение, то на общего родителя можно определить timeline-scope: --container-timeline;

timeline-scope - позволяет определить область видимости для анимаций зависящих от скролла или области просмотра;

```html
<div class="container">
  <div class="square"></div>
  <div class="content">
    <div style="height: 200vh; background-color: black"></div>
  </div>
</div>
```

```scss
.container {
  height: 100vh;
  overflow: auto;
  background-color: #100e1e;
  //добавляем ссылку на контейнер
  scroll-timeline-name: --container-timeline;
  scroll-timeline-axis: block;
  scroll-timeline: --container-timeline block;
}

.square {
  width: 100px;
  height: 100px;
  background-color: gold;
  animation: linear move;
  // завязываемся на scroll контейнера, не может быть в контейнере
  animation-timeline: --container-timeline;
  position: sticky;
  top: 0;
}

@keyframes move {
  to {
    transform: translateX(calc(100vw - 100px));
  }
}
```

Анонимные анимации

Достигаются с помощью функции scroll, куда передаем две опции

```scss
.container {
  animation: changeBg;
  // ссылка на самого себя
  animation-timeline: scroll(self);
}

.square {
  //...
  animation: linear move;
  animation-timeline: scroll(block nearest); // scroll()
  animation-range: 20% 80%; //animation-start + animation-end по умолчанию значение normal
  position: sticky;
  top: 0;
}

@keyframes move {
  to {
    transform: translateX(calc(100vw - 100px));
  }
}

@keyframes changeBg {
  to {
    background-color: coral;
  }
}
```
