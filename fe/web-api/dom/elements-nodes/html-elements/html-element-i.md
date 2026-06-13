# HTML интерфейс

[EventTarget](../../events/event-target-i.md) <- [Node](../node-i.md) <- [Element](../element-i/element-i.md) <- HTMLElement <- Все остальные html элементы (HTMLDivElement, HTMLInputElement, HTMLButtonElement)

# свойства экземпляра

- accessKey
- accessKeyLabel
- anchorElement
- attributeStyleMap
- autocapitalize
- autocorrect
- autofocus
- contentEditable
- dataset
- dir
- draggable
- editContext
- enterKeyHint
- hidden
- inert
- innerText
- inputMode
- isContentEditable
- lang
- nonce

## offsetLeft, offsetParent, offsetTop

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

## offsetHeight, offsetWidth

Содержат полную высоту, включая рамки, равен нулю если создали, но не наполнили или display:none. offsetWidth = 2border + width + 2padding

<!--  -->

- outerText
- popover
- spellcheck
- style

# style

- Это объект, который соответствует тому, что написано в атрибуте style но не в CSS!!!
- elem.style.width = "100px" работает так же как наличие в атрибуте style строки width: 100px
- Для свойств из нескольких слов:
- - background-color => elem.style.backgroundColor
- - z-index => elem.style.zIndex
- - border-left-with => elem.style.borderLeftWidth
    стили с браузерным прификсом:
    - moz-border-radius => button.style.MozBorderRadius = "5px";
    - webkit-border-radius => button.style.WebkitBorderRadius ="5px";

```js
document.body.style.backgroundColor = prompt("background color?", "green");
```

при необходимости добавить свойство стиля а позже его убрать, то можно присвоить свойству пустую строку

```js
document.body.style.display = "none"; //скрыть
setTimeout(() => (document.body.style.display = ""), 1000); //возврат к нормальному состоянию
```

div.style – это объект, доступный только для чтения, для задания нескольких стилей
style.cssText - позволяет вставить css Вставка стилей как текстовый атрибут

```js
let top = /* сложные расчёты */;
let left = /* сложные расчёты */;

// полная перезапись стилей elem, используем =
elem.style.cssText = `
  top: ${top};
  left: ${left};
`;

// добавление новых стилей к существующим стилям elem, используем +=
elem.style.cssText += `
  top: ${top};
  left: ${left};
`;

```

```html
<div id="div">Button</div>

<script>
    //перезапись флаг important
    div.style.cssText = "
      color:red !important;
      background-color: yellow;
      width: 100px;
      text-align:center
    ;"

  //добавление
    div.style.cssText += "
      color:red !important;
      background-color: yellow;
      width: 100px;
      text-align:center
    ;"

    alert(div.style.cssText); //выведет весь стиль элемента
</script>
```

При отсутствии добавления единиц измерения присвоение игнорируется

```js
document.body.style = 20; //проигнорирует
```

```js
// Функция покажет сообщение под elem, которое будет находится на фиксированной позиции, при прокрутке
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

```js
// Применение absolute

// pageY = clientY + высота прокрученной части документа (scrollTop)
// pageX = clientX + ширина горизонтально прокрученной части документа
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

<!--  -->

- tabIndex
- title
- translate
- virtualKeyboardPolicy
- writingSuggestions

# методы экземпляра

## attachInternals()

## blur()

## click()

## focus()

## hidePopover()

## showPopover()

## togglePopover()
