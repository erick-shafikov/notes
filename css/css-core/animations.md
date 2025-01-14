Анимации делятся на два типа - дискретные и вычисляемые. Дискретные меняются на 50% времени
CSS анимации легче и быстрее по сравнению с JS анимации

# transform - преобразование элемента

Позволяет растягивать, поворачивать, масштабировать элемент

```scss
.transform {
  transform: none;

  transform: matrix(1, 2, 3, 4, 5, 6);
  transform: matrix3d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
  transform: perspective(17px);
  transform: rotate(0.5turn);
  transform: rotate3d(1, 2, 3, 10deg);
  transform: rotateX(10deg);
  transform: rotateY(10deg);
  transform: rotateZ(10deg);
  transform: translate(12px, 50%);
  transform: translate3d(12px, 50%, 3em);
  transform: translateX(2em);
  transform: translateY(3in);
  transform: translateZ(2px);
  transform: scale(2, 0.5);
  transform: scale3d(2.5, 1.2, 0.3);
  transform: scaleX(2);
  transform: scaleY(0.5);
  transform: scaleZ(0.3);
  transform: skew(30deg, 20deg);
  transform: skewX(30deg);
  transform: skewY(1.07rad);

  /* Мультифункциональные значения */
  transform: translateX(10px) rotate(10deg) translateY(5px);
  transform: perspective(500px) translate(10px, 0, 20px) rotateY(3deg);
}
```

Если свойство имеет значение, отличное от none, будет создан контекст наложения. В этом случае, элемент будет действовать как содержащий блок для любых элементов position: fixed; или position: absolute; которые он содержит.

Свойство неприменимо: неизменяемые инлайновые блоки, блоки таблица-колонка, и блоки таблица-колонка-группа

!!! transform: translate Наслаиваются при анимированен одного свойства

## transform-box

определяет к чему будет приниматься трансформация

```scss
.transform-box {
  transform-box: content-box; //Поле содержимого
  transform-box: border-box; //пограничный блок
  transform-box: fill-box; //Ограничивающий блок
  transform-box: stroke-box; //Ограничивающий контур штриха
  transform-box: view-box; //Ближайший вьюпорт SVG
}
```

## transform-style

Позиционирование 3d элементов

```scss
.transform-style {
  transform-style: preserve-3d; // Показывает, что дочерний элемент должен быть спозиционирован в 3D-пространстве.
  transform-style: flat; // Показывает, что дочерний элемент лежит в той же плоскости, что и родительский.
}
```

## transform-origin

Относительно какой точки будет применяться трансформация, относительно какой координаты будет применяться трансформация, начало координат

```scss
 {
  transform-origin: 2px;
  transform-origin: bottom;

  /* x-offset | y-offset */
  transform-origin: 3cm 2px;

  /* x-offset-keyword | y-offset */
  transform-origin: left 2px;

  /* x-offset-keyword | y-offset-keyword */
  transform-origin: right top;

  /* y-offset-keyword | x-offset-keyword */
  transform-origin: top right;

  /* x-offset | y-offset | z-offset */
  transform-origin: 2px 30% 10px;

  /* x-offset-keyword | y-offset | z-offset */
  transform-origin: left 5px -3px;

  /* x-offset-keyword | y-offset-keyword | z-offset */
  transform-origin: right bottom 2cm;

  /* y-offset-keyword | x-offset-keyword | z-offset */
  transform-origin: bottom right 2cm;
}
```

Функции которые используются с transform

- [функция rotate - свойство позволяет вращать 3d объекты]
- [функция scale - позволяет растягивать объект в одном или нескольких направлениях. если принимает два значения]
- [функция translate может принимать 3 значения каждое из которых определяет ось трансформации, ненужно запомнинать в каком порядке их нужно располагать в отличие от transform]

Свойство (есть только в safari) zoom: number | % для увеличения элементов, в отличает от transform вызывает перерасчет макета

# Для настроек 3d преобразований:

### backface-visibility

будет видна или нет часть изображения в 3d, которая определена как задняя часть

```scss
.backface-visibility {
  backface-visibility: visible;
  backface-visibility: hidden;
}
```

### perspective

px расстояние от z=0 это свойство, устанавливается первое

### perspective-origin

определяет позицию с который смотрит пользователь

```scss
.perspective-origin {
  perspective-origin: x-position;

  /* Two-value syntax */
  perspective-origin: x-position y-position;

  /* When both x-position and y-position are keywords,
   the following is also valid */
  perspective-origin: y-position x-position;
}
```

### rotate

Позволяет вращать 3-d объект

```scss
.rotate {
  //* Angle value */
  rotate: 90deg;
  rotate: 0.25turn;
  rotate: 1.57rad;

  /* x, y, or z axis name plus angle */
  rotate: x 90deg;
  rotate: y 0.25turn;
  rotate: z 1.57rad;

  /* Vector plus angle value */
  rotate: 1 1 1 90deg;
}
```

### BP. Пример с кубом в 3d

```html
<!-- контейнер Определяет контейнер div, кубический div и общую грань -->
<div class="container">
  <!-- задает 3-d -->
  <div class="cube">
    <div class="face front">1</div>
    <div class="face back">2</div>
    <div class="face right">3</div>
    <div class="face left">4</div>
    <div class="face top">5</div>
    <div class="face bottom">6</div>
  </div>
</div>
```

```scss
/* Определяет контейнер div, кубический div и общую грань */
.container {
  width: 250px;
  height: 250px;
  backface-visibility: visible;
}

.cube {
  // три свойства, которые определяют трансформацию в 3d
  perspective: 550px;
  perspective-origin: 150% 150%;
  transform-style: preserve-3d;
}

.face {
  display: block;
  position: absolute;
  width: 100px;
  height: 100px;
}

/* Определяет каждое лицо на основе направления */
.front {
  background: rgba(0, 0, 0, 0.3);
  transform: translateZ(50px);
}

.back {
  background: rgba(0, 255, 0, 1);
  color: black;
  transform: rotateY(180deg) translateZ(50px);
}

.right {
  background: rgba(196, 0, 0, 0.7);
  transform: rotateY(90deg) translateZ(50px);
}

.left {
  background: rgba(0, 0, 196, 0.7);
  transform: rotateY(-90deg) translateZ(50px);
}

.top {
  background: rgba(196, 196, 0, 0.7);
  transform: rotateX(90deg) translateZ(50px);
}

.bottom {
  background: rgba(196, 0, 196, 0.7);
  transform: rotateX(-90deg) translateZ(50px);
}
```

# @keyframes создание анимации (animation-name)

Позволяет создать опорные точки анимации

свойства с !important будут проигнорированы

```scss
@keyframes slideIn {
  from {
    // 0%
    transform: translateX(0%);
  }

  to {
    // 100%
    transform: translateX(100%);
  }
}
// в процентах
@keyframes identifier {
  0% {
    top: 0;
    left: 0;
  }
  30% {
    top: 50px;
  }
  68%,
  72% {
    left: 50px;
  }
  100% {
    top: 100px;
    left: 100%;
  }
}
```

!important в keyframe будет игнорировано

```scss
//Создание анимации
@keyframes animationName {
  from {
    // начальная позиция анимации
    background-color: yellow;
    //временные функции анимации можно включать
    animation-timing-function: cubic-bezier("...");
  }
  50% {
    background-color: green;
  }
  to {
    background-color: red; // конечная позиция анимации
  }
}
```

## animation

это сокращенная запись для animation-name + animation-duration + animation-timing-function + animation-delay + animation-iteration-count + animation-direction + animation-fill-mode + animation-play-state

```scss
 {
  /* @keyframes duration | timing-function | delay |
   iteration-count | direction | fill-mode | play-state | name */
  animation: 3s ease-in 1s infinite reverse both running slidein;
}
```

## animation-name

в первую очередь задаем имя анимации

```scss
 {
  animation-name: test_05; //-specific, sliding-vertically
}
```

## animation-composition

Позволяет применять несколько анимации, полезно когда применяем два раза transform etc

```scss
 {
  animation-composition: replace; //будут перезаписываться анимации одного свойства
  animation-composition: add; // add и accumulate применяются по разному
  animation-composition: accumulate;
}
```

```scss
.square {
  height: 100px;
  width: 100px;
  background-color: gold;
  //применяем две анимации move и bounce
  animation: 2s ease-in-out infinite alternate move, 0.3s ease-in-out infinite
      alternate bounce;
  // по умолчанию
  // animation-composition: replace;
  // смешаются 2 анимации
  animation-composition: add;
  // смешаются 2 анимации как и add
  // animation-composition: accumulate;
}

@keyframes move {
  0% {
    transform: translateX(0);
  }

  100% {
    transform: translateX(calc(100vw - 140px));
  }
}

@keyframes bounce {
  0% {
    transform: translateY(0);
  }
  100% {
    transform: translateY(100px);
  }
}
```

## animation-delay

Время задержки перед стартом. При указании неправильных значений, не применится

```scss
 {
  //
  animation-delay: 1s; //через секунду
  animation-delay: -1s; //при указании отрицательных значений, анимация будет проигрываться с того времени анимации, которая указана с отрицательным значением
}
```

## animation-direction

```scss
 {
  /* Одиночная анимация */
  animation-direction: normal; //после проигрыша анимации - позиция сбросится
  animation-direction: reverse; //проигрыш задом наперед
  animation-direction: alternate; // в первом цикле normal, во втором reverse
  animation-direction: alternate-reverse; //противоположно alternate

  /* Несколько анимаций */
  animation-direction: normal, reverse;
  animation-direction: alternate, reverse, normal;

  /* Глобальные значения */
  animation-direction: inherit;
  animation-direction: initial;
  animation-direction: unset;
}
```

## animation-duration

Продолжительность анимации

```scss
 {
  animation-duration: 1s; //отрицательное и нулевое значение будет проигнорировано
}
```

## animation-fill-mode

как нужно применять стили к объекту анимации до и после проигрыша

применение начальных и конечных стилей к анимации что бы при окончании анимации применялись конечные стили без сброса стилей к изначальным

```scss
 {
  /* Ключевые слова */
  animation-fill-mode: none; //стили не будут применены до и после
  animation-fill-mode: forwards; // 100% или to в зависимости
  animation-fill-mode: backwards;
  animation-fill-mode: both;

  /* Несколько значений могут быть заданы через запятую. */
  /* Каждое значение соответствует для анимации в animation-name. */
  animation-fill-mode: none, backwards;
  animation-fill-mode: both, forwards, none;
}
```

## animation-iteration-count

```scss
 {
  animation-iteration-count: infinite; //анимация будет проигрываться бесконечно
  animation-iteration-count: 3; //3 раза
  animation-iteration-count: 2.5; //2 с половиной раза
}
```

## animation-play-state

Состояние анимации - пауза или проигрыш, если запустить анимацию после паузы она начнется с того места, где остановилась. Позволяет управлять анимацией из скрипта

```scss
 {
  animation-play-state: running; //
  animation-play-state: paused; //
}
```

## animation-timing-function

Вид временного преобразования

```scss
 {
  animation-timing-function: ease;
  animation-timing-function: ease-in;
  animation-timing-function: ease-out;
  animation-timing-function: ease-in-out;
  animation-timing-function: linear;
  animation-timing-function: step-start;
  animation-timing-function: step-end;

  // С помощью функций
  animation-timing-function: cubic-bezier(0.1, 0.7, 1, 0.1);
  animation-timing-function: steps(4, end);

  // С помощью функций шагов
  animation-timing-function: steps(4, jump-start);
  animation-timing-function: steps(10, jump-end);
  animation-timing-function: steps(20, jump-none);
  animation-timing-function: steps(5, jump-both);
  animation-timing-function: steps(6, start);
  animation-timing-function: steps(8, end);

  animation-timing-function: ease, step-start, cubic-bezier(0.1, 0.7, 1, 0.1);
}
```

# @starting-style

Позволяет определить стили для начальных стадий анимации (полезно при display: none)

определяет стартовые значения анимируемых свойств, так как при монтировании в DOM, toggle классов возвращает к значению прописанному в селекторе

```scss
@starting-style {
  //стили
}
```

```scss
p {
  animation-duration: 3s;
  animation-name: slideIn;
}

@keyframes slideIn {
  from {
    margin-left: 100%;
    width: 300%;
  }

  to {
    margin-left: 0%;
    width: 100%;
  }
}
```

События которые срабатывают на элементе при анимации

```js
var e = document.getElementById("watchme");
e.addEventListener("animationstart", listener, false);
e.addEventListener("animationend", listener, false);
e.addEventListener("animationiteration", listener, false);

e.className = "slidein";
```

# transition - добавление перехода

## transition

transition - укороченная запись для transition-property, transition-duration, transition-timing-function, и transition-delay

Значения по умолчанию:

```scss
.transition-default {
  transition-delay: 0s;
  transition-duration: 0s;
  transition-property: all;
  transition-timing-function: ease;
  transition-behavior: normal;
}
```

```scss
.transition {
  transition: margin-left 4s;
  /* имя свойства | длительность | задержка */
  transition: margin-left 4s 1s;
  /* имя свойства | длительность | временная функция | задержка */
  transition: margin-left 4s ease-in-out 1s;
  /* Применить к 2 свойствам */
  transition: margin-left 4s, color 1s;
  /* Применить ко всем изменённым свойствам */
  transition: all 0.5s ease-out;
}
```

Объединение нескольких анимаций

```css
.elementToTransition {
  /* что анимировать all – все элементы */
  transition-property: background-color, border-color;
  /* длительность анимации */
  transition-duration: 1s 2s;
  /* временная функция анимации */
  transition-timing-function: cubic-bezier() ease;
  /* Задержка анимации */
  transition-delay: 2s 0.3s;
  /* все вместе */
  transition: background-color 1s as ease 2ms, border-color 2s ease;
}
```

## transition-behavior

Позволяет запускать анимацию на дискретных свойствах. Так как анимация будет до 50% и после. Исключение display:none и visibility:hidden

```scss
 {
  transition-behavior: allow-discrete; //позволяется анимировать
  transition-behavior: normal;
}
```

## transition-delay

задержка перед анимацией, при отрицательных значений начнет проигрывать анимацию на величину значения

## transition-duration

время анимации

## transition-property

какое свойство будет анимирован, их может быть несколько

## transition-timing-function

настройка временной функции

```scss
.transition-timing-function {
  transition-timing-function: ease; //cubic-bezier(0.25, 0.1, 0.25, 1.0)
  transition-timing-function: linear; //cubic-bezier(0.0, 0.0, 1.0, 1.0)
  transition-timing-function: ease-in; //cubic-bezier(0.42, 0, 1.0, 1.0)
  transition-timing-function: ease-out; //cubic-bezier(0, 0, 0.58, 1.0)
  transition-timing-function: ease-in-out; //cubic-bezier(0.42, 0, 0.58, 1.0)
  transition-timing-function: cubic-bezier(p1, p2, p3, p4);
  // дискретные функции
  transition-timing-function: steps(n, jump-start);
  transition-timing-function: steps(n, jump-end);
  transition-timing-function: steps(n, jump-none);
  transition-timing-function: steps(n, jump-both);
  transition-timing-function: steps(n, start);
  transition-timing-function: steps(n, step-start); //jump-start.
  transition-timing-function: steps(n, step-end); // jump-end.
  transition-timing-function: step-start; //steps(1, jump-start)
  transition-timing-function: step-end; // steps(1, jump-end)
}
```

- - - [content-visibility позволяет настроить плавные анимации лоя дискретных свойств](./css-props.md/#content-visibility)

При завершении перехода срабатывает

```js
el.addEventListener("transitionend", updateTransition, true);
```

# Анимация движения по пути offset-path

Позволяет анимировать объект который следует по пути

## offset:

offset = offset-anchor + offset-distance + offset-path + offset-position + offset-rotate

позволяет определить траекторию

```scss
 {
  offset: 10px 30px;

  /* Offset path */
  offset: ray(45deg closest-side);
  offset: path("M 100 100 L 300 100 L 200 300 z");
  offset: url(arc.svg);

  /* Offset path with distance and/or rotation */
  offset: url(circle.svg) 100px;
  offset: url(circle.svg) 40%;
  offset: url(circle.svg) 30deg;
  offset: url(circle.svg) 50px 20deg;

  /* Including offset anchor */
  offset: ray(45deg closest-side) / 40px 20px;
  offset: url(arc.svg) 2cm / 0.5cm 3cm;
  offset: url(arc.svg) 30deg / 50px 100px;
}
```

### offset-anchor

Позволяет определить где будет находится элемент относительно прямой при движение по линии

```scss
 {
  offset-anchor: top;
  offset-anchor: bottom;
  offset-anchor: left;
  offset-anchor: right;
  offset-anchor: center;
  offset-anchor: auto;

  /* <percentage> values */
  offset-anchor: 25% 75%;

  /* <length> values */
  offset-anchor: 0 0;
  offset-anchor: 1cm 2cm;
  offset-anchor: 10ch 8em;

  /* Edge offsets values */
  offset-anchor: bottom 10px right 20px;
  offset-anchor: right 3em bottom 10px;
}
```

### offset-distance

px | % стартовая точка где будет находится элемент

### offset-path:

offset-distance + offset-rotate + and offset-anchor

Позволяет задать путь движения

```scss
 {
  offset-path: ray(45deg closest-side contain);
  offset-path: ray(contain 150deg at center center);
  offset-path: ray(45deg);

  /* URL */
  offset-path: url(#myCircle);

  /* Basic shape */
  offset-path: circle(50% at 25% 25%);
  offset-path: ellipse(50% 50% at 25% 25%);
  offset-path: inset(50% 50% 50% 50%);
  offset-path: polygon(30% 0%, 70% 0%, 100% 50%, 30% 100%, 0% 70%, 0% 30%);
  offset-path: path(
    "M 0,200 Q 200,200 260,80 Q 290,20 400,0 Q 300,100 400,200"
  );
  offset-path: rect(5px 5px 160px 145px round 20%);
  offset-path: xywh(0 5px 100% 75% round 15% 0);

  /* Coordinate box */
  offset-path: content-box;
  offset-path: padding-box;
  offset-path: border-box;
  offset-path: fill-box;
  offset-path: stroke-box;
  offset-path: view-box;
}
```

## offset-position

смещение относительно начала

## offset-rotate

вращение элемента относительно себя

## ray()

Отклонение от оси при создании анимации по clip-path

```scss
/* all parameters specified */
offset-path: ray(50deg closest-corner contain at 100px 20px);

/* two parameters specified, order does not matter */
offset-path: ray(contain 200deg);

/* only one parameter specified */
offset-path: ray(45deg);
```

```scss
#motion-demo {
  offset-path: path("M20,20 C20,100 200,0 200,100");
  animation: move 3000ms infinite alternate ease-in-out;
  width: 40px;
  height: 40px;
  background: cyan;
}

@keyframes move {
  0% {
    offset-distance: 0%;
  }
  100% {
    offset-distance: 100%;
  }
}
```

# scroll-driven animations (нет в ff и safari)

Существует две шкалы прогресса - прокрутка шкалы прогресса (от 0% до 100%) и временная шкала прогресса в зависимости от видимости объекта

## animation-timeline (scroll-driven-animation)

Следующие типы временных шкал могут быть установлены с помощью animation-timeline:

- ременная шкала документа по умолчанию, со старта открытия страницы
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

  /* Single animation named timeline */
  animation-timeline: --timeline_name;

  /* Single animation anonymous scroll progress timeline */
  animation-timeline: scroll();
  animation-timeline: scroll(scroller axis);

  /* Single animation anonymous view progress timeline */
  animation-timeline: view();
  animation-timeline: view(axis inset);

  /* Multiple animations */
  animation-timeline: --progressBarTimeline, --carouselTimeline;
  animation-timeline: none, --slidingTimeline;
}
```

## animation-range = animation-range-start + animation-range-end

Позволяет определить настройки срабатывания анимации, относительно начала и конце шкалы

```scss
 {
  /* single keyword or length percentage value */
  animation-range: normal; /* Equivalent to normal normal */
  animation-range: 20%; /* Equivalent to 20% normal */
  animation-range: 100px; /* Equivalent to 100px normal */

  /* single named timeline range value */
  animation-range: cover; /* Представляет полный диапазон именованной временной шкалы 0% - начал входить*/
  animation-range: contain; /* элемент полностью входит*/
  animation-range: cover 20%; /* Equivalent to cover 20% cover 100% */
  animation-range: contain 100px; /* Equivalent to contain 100px cover 100% */

  /* two values for range start and end */
  animation-range: normal 25%;
  animation-range: 25% normal;
  animation-range: 25% 50%;
  animation-range: entry exit; /* exit - начал выходить */
  animation-range: cover cover 200px; /* Equivalent to cover 0% cover 200px */
  animation-range: entry 10% exit; /* entry - начал входить */
  animation-range: 10% exit 90%;
  animation-range: entry 10% 90%;
  // entry-crossing - пересек
  // exit-crossing вышел
}
```

## scroll-timeline

для определения именованной шкалы прокрутки, сокращенная запись для scroll-timeline-name + scroll-timeline-axis

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

## функция scroll()

Функция для отслеживания временной шкалы анонимной анимации зависящей от скролла

```scss
 {
  animation-timeline: scroll();

  /* Values for selecting the scroller element */
  animation-timeline: scroll(nearest); /* Default */
  animation-timeline: scroll(root);
  animation-timeline: scroll(self);

  /* Values for selecting the axis */
  animation-timeline: scroll(block); /* Default */
  animation-timeline: scroll(inline);
  animation-timeline: scroll(y);
  animation-timeline: scroll(x);

  /* Examples that specify scroller and axis */
  animation-timeline: scroll(block nearest); /* Default */
  animation-timeline: scroll(inline root);
  animation-timeline: scroll(x self);
}
```

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

## view-timeline-inset

Корректирует срабатывание анимации относительно скролла

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

# view-port animation при попадании в поле зрения (-ff, -safari)

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
  //  задаем контейнер !!!TODOMDN
  scroll-timeline-name: --container-timeline;
  //  направление !!!TODOMDN
  scroll-timeline-axis: block;
  //  shorthand !!!TODOMDN
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
  //...
  animation: changeBg;
  animation-timeline: scroll(self);
}

.square {
  //...
  animation: linear move;
  animation-timeline: scroll(block nearest); //=== scroll()
  animation-timeline: scroll(block root); // от корня
  animation-timeline: scroll(block self); // от себя
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

Анимация, которая основывается на попадании элемента в область видимости

# @view-transition (-ff, -safari)

```scss
@view-transition {
  navigation: auto;
  navigation: none; // Документ не будет подвергнут переходу вида.
}
```

## view-timeline

view-timeline-name + view-timeline-axis

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

## JS обработка анимаций

```js
animateElement.addEventListener("transitioned", function () {
  //выполнить что-либо по завершению анимации
});
```

# оптимизация will-change

Возможность уведомить браузер о том, что что-то изменится на странице

```scss
.will-change {
  will-change: auto;
  will-change: scroll-position; //Указывает, что автор ожидает анимацию или изменение положения скролла элемента в ближайшем будущем.
  will-change: contents; //Указывает, что автор ожидает анимацию или изменение чего то в контенте элемента в ближайшем будущем.
  will-change: transform; /* Example of <custom-ident> */
  will-change: opacity; /* Example of <custom-ident> */
  will-change: left, top; /* Example of two <animateable-feature> */
}
```

!!!Не применяйте will-change к большому числу элементов.

```scss
.sidebar {
  will-change: transform;
}
```

```js
var el = document.getElementById("element");

// Set will-change when the element is hovered
el.addEventListener("mouseenter", hintBrowser);
el.addEventListener("animationEnd", removeHint);

function hintBrowser() {
  // The optimizable properties that are going to change
  // in the animation's keyframes block
  this.style.willChange = "transform, opacity";
}

function removeHint() {
  this.style.willChange = "auto";
}
```
