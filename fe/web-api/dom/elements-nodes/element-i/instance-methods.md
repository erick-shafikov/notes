методы экземпляра

# after(node1, node2, nodeN)

вставляет после Element

# animate(keyframes, options)

создает и запускает анимацию

- - keyframes - массив объектов
- - options:
- - - id
- - - rangeEnd

```js
const newspaperSpinning = [
  { transform: "rotate(0) scale(1)" },
  { transform: "rotate(360deg) scale(0)" },
];

const newspaperTiming = {
  duration: 2000,
  iterations: 1,
};

const newspaper = document.querySelector(".newspaper");

newspaper.addEventListener("click", () => {
  newspaper.animate(newspaperSpinning, newspaperTiming);
});
```

- append() - вставляет в конец Element.
- attachShadow() - добавляет теневое DOM дерево к Element
- before(param1, param2, paramN) - добавляет в начало Element
- checkVisibility(options) ⇒ boolean виден ли элемент (display:none)
- - options:
- - - contentVisibilityAuto
- - - opacityProperty
- - - visibilityProperty
- - - checkOpacity
- - - checkVisibilityCSS
- closest(selectors) ⇒ возвращает ближайший родительский компонент по селектору
- computedStyleMap() ⇒ StylePropertyMapReadOnly
- getAnimations() ⇒ Animation
- getAttribute(attributeName) ⇒ значение атрибута по имени
- getAttributeNames() ⇒ массив с именами атрибутов
- getAttributeNode() ⇒ Attr
- getAttributeNodeNS() ⇒ Attr для NS
- getAttributeNS() ⇒ Attr для NS

<!--  -->

# getBoundingClientRect() ⇒ размер элемента и его расположение left, top, right, bottom, x, y, width и height

getBoundingClientRect() – возвращает координаты в контексте окна для минимального по размеру прямоугольника, который заключает в себе элемент elem в виде объекта встроенного класса DOMRect

x/y – координаты начла прямоугольника относительно окна
width/height – ширина высота прямоугольника
top/bottom – Y верхней нижней границы прямоугольника bottom = y + height
left/right – X координата левой правой границы прямоугольника right = y + height

в момент когда элемент уходит за область прокрутки, то возвращаются отрицательные координаты

<!--  -->

- getClientRects() ⇒ DOMRect[]
- getElementsByClassName(names) ⇒ HTMLCollection из потомков с names в качестве classNames
- getElementsByTagName() ⇒ HTMLCollection из потомков по имя тега
- getElementsByTagNameNS() ⇒ HTMLCollection из потомков по имя тега дял NS
- getHTML(options) ⇒ элементы DOM в виде строки
- hasAttribute(name) ⇒ boolean если ли данный атрибут на Element
- hasAttributeNS()
- hasAttributes() ⇒ boolean если ли атрибуты
- hasPointerCapture(pointerId)
- insertAdjacentElement(position, element) - добавит Element в dom дерево
- - position:
- - - beforebegin
- - - afterbegin
- - - beforeend
- - - afterend
- insertAdjacentHTML(position, text) - вставит text как html
- insertAdjacentText(position, element)- вставит text как html
- matches(selectorString) - соответствует ли Element css селектору selectorString
- moveBefore(movedNode, referenceNode) - movedNode переместит перед referenceNode
- prepend(param1, param2, paramN) - вставит перед концом
- querySelector(selectors) ⇒ первый Element по selectors
- querySelectorAll(selectors) ⇒ статичный NodeList
- releasePointerCapture() - остановит Pointer Capture
- remove() - удалит элемент из dom
- removeAttribute(attrName) - удалит атрибут с Element
- removeAttributeNode() - удалит Attr с Element
- removeAttributeNS()
- replaceChildren(param1, param2, paramN) - заменит Element элементами param1, param2, paramN
- replaceWith(param1, param2, paramN) - заменит детей Element
- requestFullscreen() - сделать Element на весь экран
- requestPointerLock() - блокировка курсора
- scroll(xCoord, yCoord) - прокрутить до (xCoord, yCoord)
- scroll(options) - прокрутить с учетом behavior
- - options:
- - - behavior: smooth, instant, auto
- scrollBy(xCoord, yCoord) или scrollBy(options)
- scrollIntoView(alignToTop) - элемент в поле видемости alignToTop - будет ли верхняя граница по верху
- scrollIntoView(scrollIntoViewOptions) -
- - scrollIntoViewOptionsЖ
- - - behavior
- - - block
- - - inline
- scrollIntoViewIfNeeded()Не стандартно
- scrollTo(x-coord, y-coord) - прокрутка внутри данного элемента
- setAttribute(name, value) - установить атрибут name значение value
- setAttributeNode(attribute) - attribute - Attr
- setAttributeNodeNS()
- setAttributeNS()
- setCapture()Не стандартноУстарело
- setHTMLUnsafe() - для преобразования строки в HTML
- setPointerCapture() - сделать элемент target для Pointer Events
- toggleAttribute(name, force) - для boolean атрибутов
