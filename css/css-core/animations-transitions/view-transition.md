# view-timeline (на контейнере)

view-timeline = view-timeline-name + view-timeline-axis

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

# animation-timeline (на элементе с анимацией)

## view()

Функция для отслеживания временной шкалы анонимной анимации зависящей от видимости элемента от скролла

```scss
.view-function {
  /* Function with no parameters set */
  animation-timeline: view();

  /* Values for selecting the axis */
  animation-timeline: view(block); /* Default */
  animation-timeline: view(inline);
  animation-timeline: view(y);
  animation-timeline: view(x);

  /* Values for the inset */
  animation-timeline: view(auto); /* Default */
  animation-timeline: view(20%);
  animation-timeline: view(200px);
  animation-timeline: view(20% 40%);
  animation-timeline: view(20% 200px);
  animation-timeline: view(100px 200px);
  animation-timeline: view(auto 200px);

  /* Examples that specify axis and inset */
  animation-timeline: view(block auto); /* Default */
  animation-timeline: view(inline 20%);
  animation-timeline: view(x 200px auto);
}
```

## animation-range

animation-range = animation-range-start + animation-range-end

Позволяет определить настройки срабатывания анимации, относительно начала и конце шкалы

```scss
.animation-range {
  animation-range: normal; // Значение по умолчанию
  animation-range: 20%; // 20% от начала
  animation-range: 100px; // 100px от начала

  animation-range: cover; // Представляет полный диапазон именованной временной шкалы 0% - начал входить
  animation-range: contain; // настройка для ситуаций, если элемент больше view-port
  //Если объектный элемент меньше области прокрутки, охватывается областью прокрутки (прогресс 0%), до не полностью охватывается областью прокрутки (прогресс 100%).
  //Если объектный элемент больше области прокрутки, от точки впервые полностью перекрывает область прокрутки (прогресс 0%), до точки больше не полностью перекрывает область прокрутки (прогресс 100%).
  animation-range: cover 20%; //cover 20% cover 100%
  animation-range: contain 100px; //100px cover 100%

  // two values for range start and end
  animation-range: normal 25%;
  animation-range: 25% normal;
  animation-range: 25% 50%;
  animation-range: entry exit; // exit - начал выходить
  animation-range: cover cover 200px; // Equivalent to cover 0% cover 200px
  animation-range: entry 10% exit; // entry - начал входить
  animation-range: 10% exit 90%;
  animation-range: entry 10% 90%;
  // entry-crossing - пересек
  // exit-crossing вышел
}
```

# view-timeline-inset

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

# BPs

## BP. пример

```html
<!-- контейнер 1.1[3]-->
<div class="container">
  <div class="content">
    <!-- для создания контента, который выходит за пределы видимости -->
    <div style="height: 200vh; background-color: #2d2337"></div>
    <div class="progress">
      <!-- внутренний элемент который будет анимирован при  скролле 2.1[3]-->
      <div class="progress-inner"></div>
    </div>
    <!-- для создания контента, который выходит за пределы видимости -->
    <div style="height: 200vh; background-color: #2d2337"></div>
  </div>
</div>
```

```scss
body {
  // если нужно анимировать элемент который выше по иерархии 1.2[3]
  timeline-scope: --progress-view;
}

.container {
  height: 100vh;
  overflow: auto;
  background-color: #100e1e;
  // анимация на внешний контейнер 1.3[3]
  animation: both changeBg;
  animation-timeline: --progress-view;
}

.content {
  padding: 30px;
  max-width: 700px;
  margin: 0 auto;
}

.progress {
  width: 100%;
  height: 100px;
  background-color: rgb(57, 41, 67);
  position: relative;
  margin: 20px 0;
}
.progress-inner {
  position: absolute;
  width: 50%;
  height: 100%;
  top: 0;
  left: 0;
  background-color: rgb(209, 90, 213);
  // определяем timeline 2.2[3]
  view-timeline: --progress-view;
  animation: animateWidth linear both;
  // связываем анимацию с timeline
  animation-timeline: --progress-view;
  /* переместить линию срабатывания начала анимации */
  /* animation-range: 30% 70%; */
  /* view-timeline-inset задает рамку срабатывания, если заданы отрицательные значения - 
  сработает за пределами. Могут быть заданы  
  animation-range и view-timeline-inset тогда смещения будут суммироваться
  */
  view-timeline-inset: 20% 10%;
  // свойства которые позволяют управлять пересечением
  animation-range-start: normal; /*cover | contain - полностью в области видимости */
  /* exit - если элемент больше вп, entry-crossing, entry, exit-crossing */
  animation-range-end: normal; /*cover | contain*/
}

// анимация внутреннего элемента 1.4[4]
@keyframes animateWidth {
  0% {
    width: 0%;
  }
  100% {
    width: 100%;
  }
}

// анимация внешнего элемента 2.3[3]
@keyframes changeBg {
  to {
    background-color: rgb(32, 21, 82);
  }
}
```

Анонимный вариант

```scss
.container {
  //
  animation: both changeBg;
  animation-timeline: view();
}

.content {
  //
}

.progress {
  //
}
.progress-inner {
  //
  animation: linear both animateWidth;
  animation-timeline: view(20% 30%);
}
```
