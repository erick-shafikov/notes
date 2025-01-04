# Анимации и трансформации элементов

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

# transform-box

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

# perspective-origin

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

Пример с кубом в 3d

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

# @keyframes создание анимации

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

- [свойство animation = animation-name + animation-duration + animation-timing-function + animation-delay + animation-iteration-count + animation-direction + animation-fill-mode + animation-play-state](./css-props.md/#animation)
- - [animation-name имя анимации](./css-props.md#animation-name)
- - [animation-composition свойство для смешивания двух анимаций](./css-props.md/#animation-composition)
- - [animation-delay задержка анимации](./css-props.md#animation-delay)
- - [animation-direction направленность анимации](./css-props.md#animation-direction)
- - [animation-duration длительность анимации](./css-props.md#animation-duration)
- - [animation-fill-mode применение начальных и конечных стилей к анимации что бы при окончании анимации применялись конечные стили без сброса стилей к изначальным](./css-props.md#animation-fill-mode)
- - [animation-iteration-count количество повторов анимации](./css-props.md#animation-iteration-count)
- - [animation-play-state воспроизведение и остановка](./css-props.md#animation-play-state)
- - [animation-timing-function временная кривая для анимации](./css-props.md#animation-timing-function)
- свойства transition
- - [transition-behavior позволяет задать характер анимации позволяет запускать анимации для дискретных свойств](./css-props.md/#transition-behavior)
- [@starting-style определяет стартовые значения анимируемых свойств, так как при монтировании в DOM, toggle классов возвращает к значению прописанному в селекторе](./at-rules.md/#starting-style)
- временные функции анимации
- - [!!!TODO_MDN linear]
- - [!!!TODO_MDN cubic-bezier]
- - [!!!TODO_MDN step]

- [@keyframes - создает шаги анимации, можно использовать проценты или ключевые слова from to](./at-rules.md#keyframes)

!!! transform: translate Наслаиваются при анимированен одного свойства

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

### transition-behavior

Позволяет запускать анимацию на дискретных свойствах. Так как анимация будет до 50% и после. Исключение display:none и visibility:hidden

```scss
 {
  transition-behavior: allow-discrete; //позволяется анимировать
  transition-behavior: normal;
}
```

- - transition-delay - задержка перед анимацией, при отрицательных значений начнет проигрывать анимацию на величину значения
- - transition-duration - время анимации
- - transition-property какое свойство будет анимирован, их может быть несколько

# transition-timing-function

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

- [свойство offset позволяет определить траекторию](./css-props.md/#offset)
- [расположение элемента относительно прямой движения](./css-props.md/#offset-anchor)
- [offset-distance: px | % стартовая точка где будет находится элемент]
- [offset-path = offset-distance + offset-rotate + offset-anchor](./css-props.md/#offset-path)
- [offset-position смещение относительно начала]
- [offset-rotate вращение элемента относительно себя]
- [ray функция для вращения](./functions.md#ray)

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

# scroll-driven animations

Существует две шкалы прогресса - прокрутка шкалы прогресса (от 0% до 100%) и временная шкала прогресса в зависимости от видимости объекта

- (нет в ff и safari)[animation-timeline свойство определяет временную шкалу для анимации](./css-props.md/#animation-timeline)
- (нет в ff и safari)[animation-range позволяет управлять срабатыванием анимации](./css-props.md/#animation-range--animation-range-start--animation-range-end)
- (нет в ff и safari)[scroll-timeline для определения именованной шкалы прокрутки, сокращенная запись для scroll-timeline-name + scroll-timeline-axis]
- [scroll() Функция для отслеживания временной шкалы анонимной анимации зависящей от скролла](./functions.md/#scroll-scroll-driven-animation)

## view-port animation

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

### Анимация от вью порта, при попадании в поле зрения

Анимация, которая основывается на попадании элемента в область видимости

- [(нет в ff и safari)view-timeline = view-timeline-name + view-timeline-axis](./css-props.md/#view-timeline--view-timeline-name--view-timeline-axis)
- [view-transition-name: nameOfTViewTransition | none позволяет отключить/включить ]

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

## view transitions

определяет свойство View Transition API (!!!TODO)

## оптимизация will-change

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
