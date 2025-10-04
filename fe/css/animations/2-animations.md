Анимации делятся на два типа - дискретные и вычисляемые. Дискретные меняются на 50% времени

- CSS анимации легче и быстрее по сравнению с JS анимации
- по возможности анимировать только transform, opacity, filter, backdrop-filter
- выносить анимации дальше по z-index

<!-- transition ------------------------------------------------------------------------------------------------------------------------------>

# transition: - добавление перехода

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

## transition-property

какое свойство будет анимирован, их может быть несколько

## transition-duration

время анимации

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

## transition-delay

задержка перед анимацией, при отрицательных значений начнет проигрывать анимацию на величину значения

## transition-behavior

Позволяет запускать анимацию на дискретных свойствах. Так как анимация будет до 50% и после. Исключение display:none и visibility:hidden

```scss
 {
  transition-behavior: allow-discrete; //позволяется анимировать
  transition-behavior: normal;
}
```

<!-- animation ------------------------------------------------------------------------------------------------------------------------------->

# animation

Тип анимаций, которые идут в паре с keyframe

это сокращенная запись для animation-name + animation-duration + animation-timing-function + animation-delay + animation-iteration-count + animation-direction + animation-fill-mode + animation-play-state

```scss
.animation {
  /* @keyframes duration | timing-function | delay |
   iteration-count | direction | fill-mode | play-state | name */
  animation: 3s ease-in 1s infinite reverse both running slidein;
  // начальные значения
  animation-name: none;
  animation-duration: 0s;
  animation-timing-function: ease;
  animation-delay: 0s;
  animation-iteration-count: 1;
  animation-direction: normal;
  animation-fill-mode: none;
  animation-play-state: running;
  animation-timeline: auto;
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

Позволяет применять несколько анимации на один элемент

```scss
.animation-composition {
  animation-composition: replace; //будут перезаписываться анимации одного свойства
  animation-composition: add; // Применяется сумма изменений
  animation-composition: accumulate; // Применяется сумма изменений

  //множественное применение
  animation-composition: replace, add;
  animation-composition: add, accumulate;
  animation-composition: replace, add, accumulate;
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

## interpolate-size (-ff -safari)

Позволяет интерполировать процентные значения ширины и высоты в пиксели

```scss
.interpolate-size {
  interpolate-size: allow-keywords;
  interpolate-size: numeric-only;
}
```

Пример

```scss
:root {
  // наследуется
  interpolate-size: allow-keywords;
}
```

```scss
section {
  height: 2.5rem;
  overflow: hidden;
  // позволит плавно анимировать
  // без этого свойства анимация не проигрывается
  transition: height ease 1s;
}

section:hover,
section:focus {
  height: max-content;
}
```

## @keyframes

Позволяет создать опорные точки анимации

свойства с !important будут проигнорированы

```scss
// вариант с from - to
// animation-name: slideIn
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

## @starting-style

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

# оптимизация

## content-visibility

[content-visibility позволяет настроить плавные анимации лоя дискретных свойств](../containment.md#content-visibility)

При завершении перехода срабатывает

```js
el.addEventListener("transitionend", updateTransition, true);
```

## JS обработка анимаций

```js
animateElement.addEventListener("transitioned", function () {
  //выполнить что-либо по завершению анимации
});
```

## will-change

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

## Влияние свойств на анимацию

```scss
.awesome-block {
  width: 2rem;
  height: 2rem;
  background-color: lightblue;
  position: absolute;
  animation: move 2s infinite;
}

//в этом случае перерисовки не будет
@keyframes move {
  0% {
    translate: 0;
  }

  100% {
    translate: 2rem;
  }
}

//браузеры будут перерисовывать элемент на каждое изменение значения
@keyframes move {
  0% {
    left: 0;
  }

  100% {
    left: 2rem;
  }
}
```

через calc size

```scss
.card {
  height: 0;
}

.card.open {
  height: calc-size(auto);
}
```
