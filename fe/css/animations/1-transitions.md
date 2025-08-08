# transform - преобразование элемента

Позволяет растягивать, поворачивать, масштабировать элемент

```scss
.transform {
  transform: none;
  //
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

его следует устанавливать для всех не прямых потомков элемента.

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
- [функция translate может принимать 3 значения каждое из которых определяет ось трансформации, ненужно запоминать в каком порядке их нужно располагать в отличие от transform]

Свойство (есть только в safari) zoom: number | % для увеличения элементов, в отличает от transform вызывает перерасчет макета

<!-- Альтернативы transform ---------------------------------------------------------------------------------------------------------->

## Альтернативы transform:

### translate

Позволяет быстро поменять расположение элемента либо в 2d или 3d

```scss
.translate {
  // преобразует в 2d пространстве, третье значение для оси z
  translate: 1px 50%;
}
```

### scale

Позволяет без transform растянуть объект по осям

```scss
.scale {
  scale: 2;
  scale: 50%;
  scale: 2 0.5;
  scale: 200% 50% 200%;
}
```

### zoom

Позволяет без transform растянуть объект по осям

## функции для свойства transform

- matrix() matrix3d() - преобразование в 2d и 3d
- perspective() - создание перспективы
- rotate() - вращение
- rotate3d() - вращение
- rotateX() - вращение по горизонтальной оси
- rotateY() - вращение по вертикальной оси
- rotateZ() - по перпендикулярной
- scale() - растягивание
- scale3d() - растягивание
- - scaleX() - растягивание
- - scaleY() - растягивание
- - scaleZ() - растягивание
- skew() - скашивание
- - skewX() - скашивание
- - skewY() - скашивание
- translate() - перемещение по плоскости
- - translate3d()
- - translateX()
- - translateY()
- - translateZ()

<!-- Для настроек 3d преобразований ---------------------------------------------------------------------------------------------------------->

# Для настроек 3d преобразований:

## backface-visibility

будет видна или нет часть изображения в 3d, которая определена как задняя часть

```scss
.backface-visibility {
  backface-visibility: visible;
  backface-visibility: hidden;
}
```

## perspective

px расстояние от z=0 это свойство, устанавливается первое

## perspective-origin

определяет позицию с который смотрит пользователь

```scss
.perspective-origin {
  perspective-origin: x-position; // left === 0% | center === 50% | right === 100%
  perspective-origin: y-position; // top | center | bottom

  perspective-origin: x-position y-position;

  // When both x-position and y-position are keywords, the following is also valid
  perspective-origin: y-position x-position;
}
```

## rotate

Позволяет вращать 3-d объект

```scss
.rotate {
  // относительно центра
  rotate: 90deg;
  rotate: 0.25turn;
  rotate: 1.57rad;

  // по определенной оси
  rotate: x 90deg;
  rotate: y 0.25turn;
  rotate: z 1.57rad;

  /* вращение в 3d */
  rotate: 1 1 1 90deg;
}
```

## BP. Пример с кубом в 3d

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

# bp. анимация display:none

решение opacity + height или width

```scss
li {
  height: 50px; /* any measurable value, not "auto" */
  opacity: 1;
  transition: height 0ms 0ms, opacity 400ms 0ms;
}

.fade-out {
  overflow: hidden; /* Hide the element content, while height = 0 */
  height: 0;
  opacity: 0;
  padding: 0;
  transition: height 0ms 400ms, padding 0ms 400ms, opacity 400ms 0ms;
}
// или через max-height
.collapsible {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.4s ease;
}

.collapsible.open {
  max-height: 500px;
}
```

через js

```js
document.getElementById("fadeButton").addEventListener("click", function () {
  const element = document.getElementById("myElement");
  element.style.opacity = "0";
  setTimeout(() => {
    element.style.display = "none";
  }, 1000); // Match this value with the duration in CSS
});

document.getElementById("toggleButton").addEventListener("click", function () {
  const el = document.querySelector(".expandable");
  const contentHeight = el.scrollHeight;
  el.style.height = contentHeight + "px";

  el.addEventListener("transitionend", function (e) {
    el.removeEventListener("transitionend", arguments.callee);
    el.style.height = "auto";
  });
});
```

Вариант 3 display и content-visibility

```scss
.fade-out {
  animation: fade-out 0.25s forwards;
}

/* Keyframe animations */
@keyframes fade-out {
  100% {
    opacity: 0;
    display: none;
  }
}
```

Вариант 4 transition-behavior: allow-discrete

```scss
.card {
  @starting-style {
    opacity: 0;
  }

  opacity: 1;
  transition: opacity 0.5s;

  transition: opacity 0.5s, display 0.5s;
  transition-behavior: allow-discrete; /* this is essential */
}

.card.fade-out {
  opacity: 0;
  display: none;
}
```
