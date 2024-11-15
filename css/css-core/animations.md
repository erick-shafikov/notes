# Анимации и трансформации элементов

Анимации делятся на два типа - дискретные и вычисляемые. Дискретные меняются на 50% времени
CSS анимации легче и быстрее по сравнению с JS анимации

## преобразование элемента (transform)

[transform свойство по перемещению и масштабированию элементов](./css-props.md/#transform)

- - [transform-box трансформация и рамки](./css-props.md/#transform-box)
- - [transform-style для пространственных преобразований](./css-props.md/#transform-style)
- - [transform-origin - относительно какой координаты будет применяться трансформация, начало координат](./css-props.md/#transform-origin)

Функции которые используются с transform

- [функция rotate - свойство позволяет вращать 3d объекты]
- [функция scale - позволяет растягивать объект в одном или нескольких направлениях. если принимает два значения]
- [функция translate может принимать 3 значения каждое из которых определяет ось трансформации, ненужно запомнинать в каком порядке их нужно располагать в отличие от transform]

Свойство (есть только в safari) zoom: number | % для увеличения элементов, в отличает от transform вызывает перерасчет макета

## Для настроек 3d анимаций:

- [backface-visibility: visible | hidden позволяют скрыть заднюю грань элемента при 3d трансформаций](./css-props.md/#backface-visibility)
- [perspective: px расстояние от z=0 это свойство, устанавливается первое]
- - [perspective-origin определяет позицию с который смотрит пользователь](./css-props.md/#perspective-origin)

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

## добавление перехода (transition)

- [transition укороченная запись для transition-property + transition-duration + transition-timing-function + transition-delay](./css-props.md/#transition)
- - transition-delay - задержка перед анимацией, при отрицательных значений начнет проигрывать анимацию на величину значения
- - transition-duration - время анимации
- - transition-property какое свойство будет анимирован, их может быть несколько
- - [transition-timing-function - временная функция](./css-props.md/#transition-timing-function)
- - [transition-behavior - поведение для анимации, позволяет настроить дискретные анимации](./css-props.md#transition-behavior)
- - - [content-visibility позволяет настроить плавные анимации лоя дискретных свойств](./css-props.md/#content-visibility)

## создание анимации (keyframes)

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
- - [!!!TODOMDN linear]
- - [!!!TODOMDN cubic-bezier]
- - [!!!TODOMDN step]

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

## Анимация движения по пути offset-path

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

## scroll-driven animations

Существует две шкалы прогресса - прокрутка шкалы прогресса (от 0% до 100%) и временная шкала прогресса в зависимости от видимости объекта

- (нет в ff и safari)[animation-timeline свойство определяет временную шкалу для анимации](./css-props.md/#animation-timeline)
- (нет в ff и safari)[animation-range позволяет управлять срабатыванием анимации](./css-props.md/#animation-range--animation-range-start--animation-range-end)
- (нет в ff и safari)[scroll-timeline для определения именованной шкалы прокрутки, сокращенная запись для scroll-timeline-name + scroll-timeline-axis]
- [scroll() Функция для отслеживания временной шкалы анонимной анимации зависящей от скролла](./functions.md/#scroll-scroll-driven-animation)

### Анимация от вью порта

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
