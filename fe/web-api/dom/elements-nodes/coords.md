# прокрутка scrollTo scrollBy scrollView

# Координаты

- clientX, clientY - координаты от видимой области
- pageX, pageY - координаты от начала страницы
- screenX, screenY - координаты от начала экрана

при position: fixed – отсчет от верхнего левого угла окна (window) clientX, clientY
MouseEvent.clientX MouseEvent.clientY - указывают на положение курсора
при position: absolute – отсчет от верхнего левого ула документа pageX, pageY

при прокрутке clientY меняется pageY

# Применение для fixed

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

# Применение absolute

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
