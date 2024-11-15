# Метрики

## Расстояние до родителя offsetParent, offsetLeft, offsetTop

- offsetParent, offsetLeft, offsetTop что-то типа margin, расстояние до ближайшего предка, до элемента, который является CSS позиционированным (position: absolute/relative/fixed/sticky) или td, th, table или body. offsetParent == null если для скрытых элементов (display:none) для элементов body и html и для элементов с позицией fixed

```html
<main style = "position: relative" id="main">
  <article>
    <div id="example" style="position: absolute; left: 180px; top: 180px"></div>
  </article>
</main>
<script>
  alert(example.offsetParent.id); //main
  alert(example.offsetLeft); //180
  alert(example.offsetTop); //180
<script>
```

## Размер элемента offsetWidth, offsetHeight

Содержат полную высоту, включая рамки, равен нулю если создали, но не наполнили или display:none. offsetWidth = 2border + width + 2padding

## clientTop, clientLeft (border)

для метрик border выражает ширину рамки, но на самом деле это отступы внутренней части элемента от внешней

## clientHight, clientWidth (содержимое + padding)

размер области внутри рамок элемента, если нет внутренних отступов padding то clientWidth/height в точности равны размеру содержимого

## scrollWidth, scrollHeight

scrollWidth, scrollHeight - свойства как clientWidth и clientHeight включающие в себя прокрученную область, которую не видно

## scrollLeft, scrollTop

прокрученной в данный момент части элемента, ScrollTop – то то, сколько прокручено сверху scrollTop на 0 или на infinity – прокрутить вверх или до конца

распахнуть элемент на всю высоту

```js
element.style.height = `${element.scrollHeight}px`;
```

Проверка на видимость

```js
function isHidden(elem) {
  //вернет true для элементов которые показываются, но их размер равен нулю
  return !elem.offsetWidth && !elem.offsetHight;
}
```

В чем отличие между CSS св-ва width и client Width

clientWidth возвращает число, а getComputedStyle(elem).width – строку с px на конце.
getComputedStyle не всегда даст ширину, он может вернуть, к примеру, "auto" для строчного элемента.

clientWidth соответствует внутренней области элемента, включая внутренние отступы padding, а CSS-ширина (при стандартном значении box-sizing) соответствует внутренней области без внутренних отступов padding.

Если есть полоса прокрутки, и для неё зарезервировано место, то некоторые браузеры вычитают его из CSS-ширины (т.к. оно больше недоступно для содержимого), а некоторые – нет. Свойство clientWidth всегда ведёт себя одинаково: оно всегда обозначает размер за вычетом прокрутки, т.е. реально доступный для содержимого

# Размер и прокрутка окна

## Ширина и высота документа

Что бы получить размеры окна можно взять свойства document.Element.clientHeight, document.Element.clientWidth

в отличие от window.innerHight и window.innerWidth – указывают на видимую часть документа , то есть и на часть с прокруткой
из-за отличий в браузерах следует брать максимальное значение

```js
let scrollHeight = Math.max(
  document.body.scrollHeight,
  document.documentElement.scrollHeight,
  document.body.offsetHeight,
  document.documentElement.offsetHeight,
  document.body.clientHeight,
  document.documentElement.clientHeight
);
```

## Получение текущей позиции

window.pageYOffset === window.scrollX, window.pageXOffset === window.scrollY

## прокрутка scrollTo scrollBy scrollView

```js
window.scrollBy(x, y); //– прокручивает страницу относительно ее текущего положения
window.scrollTo(pageX, pageY); //– прокручивает страницу на абсолютные координаты, чтобы прокрутить в самое начало window.scrollTo(0,0)
//или с настройками
window.scrollTo({
  top: 100,
  left: 0,
  behavior: "smooth",
});
elem.scrollIntoView(true); //– прокручивает страницу так, чтобы элемент оказался сверху при top = true и внизу если top = false
// или с настройками
this.scrollIntoView({
  behavior: "smooth",
  block: "end",
  inline: "nearest",
});
document.body.style.overflow = "hidden"; //выключает прокрутку
document.body.style.overflow = ""; //включает ее обратно
```

# Координаты

при position: fixed – отсчет от верхнего левого угла окна clientX, clientY
при position: absolute – отсчет от верхнего левого ула документа pageX, pageY

при прокрутке clientY меняется pageY

## getBoundingClientRect

elem. getBoundingClientRect() – возвращает координаты в контексте окна для минимального по размеру прямоугольника, который заключает в себе элемент elem в виде объекта встроенного класса DOMRect

x/y – координаты начла прямоугольника относительно окна
width/height – ширина высота прямоугольника
top/bottom – Y верхней нижней границы прямоугольника bottom = y + height
left/right – X координата левой правой границы прямоугольника right = y + height

в момент когда элемент уходит за область прокрутки, то возвращаются отрицательные координаты

## elementFromPoint(x, y)

возвращает самый глубоко вложенный элементы окне находящийся по координатам (x,y)

### Применение для fixed

Функция покажет сообщение под elem, которое будет находится на фиксированной позиции, при прокрутке

```js
let elem = document.getElementById("coords-show-mark");

function createMessageUnder(elem, html) {
  let message = document.createElement("div"); //создаем элемент который будет содержать сообщение
  message.style.cssText = "position:fixed; color:red"; //добавим css стиль, из-за то, что Position:fixed то сообщение будет находится на одном месте даже при прокрутке
  let coords = elem.getBoundingClientRect(); //устанавливаем координаты

  message.style.left = coords.left + "px"; //найдем координаты элемента
  message.style.top = coords.bottom + "px";

  message.innerHTML = html;

  return message;
}

let message = createMessageUnder(elem, "Hello");
document.body.append(message);
setTimeout(() => message.remove(), 5000);
```

## Применение absolute

pageY = clientY + высота прокрученной части документа (scrollTop)
pageX = clientX + ширина горизонтально прокрученной части документа

```js
function getCoords(elem) {
  let box = elem.getBoundingClientRect();

  return {
    top: box.top + pageYOffset,
    left: box.left + pageYOffset,
    bottom: box.bottom + window.pageYOffset,
    left: box.left + window.pageXOffset,
  };
}

function createMessageUnder(elem, html) {
  let message = document.createElement("div");
  message.style.cssText = "position:absolute; color:red";
  let coords = getCoords(elem);

  message.style.left = coords.left + "px";
  message.style.top = coords.bottom + "px";

  message.innerHTML = html;

  return message;
}
```
