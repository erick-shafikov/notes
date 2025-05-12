# Element

наследуются от [Node](./node-intreface.md)

все html элементы (HTMLElement) наследуются от Element

## свойства экземпляра

- aria-свойства: ariaActiveDescendantElement, ariaAtomic, ariaAutoComplete, ariaBrailleLabel, ariaBrailleRoleDescription, ariaBusy, ariaChecked, ariaColCount, ariaColIndex, ariaColIndexText, ariaColSpan, ariaControlsElements, ariaCurrent, ariaDescribedByElements, ariaDescription, ariaDetailsElements, ariaDisabled, ariaErrorMessageElements, ariaExpanded, ariaFlowToElements, ariaHasPopup, ariaHidden, ariaInvalid, ariaKeyShortcuts, ariaLabel, ariaLabelledByElements, ariaLevel, ariaLive, ariaModal, ariaMultiLine, ariaMultiSelectable, ariaOrientation, ariaOwnsElements, ariaPlaceholder, ariaPosInSet, ariaPressed, ariaReadOnly, ariaRelevantНе стандартно, ariaRequired, ariaRoleDescription, ariaRowCount, ariaRowIndex, ariaRowIndexText, ariaRowSpan, ariaSelected, ariaSetSize, ariaSort, ariaValueMax, ariaValueMin, ariaValueNow, ariaValueText

- assignedSlot ⇒ HTMLSlotElement, если есть shadow-root элементы
- attributes ⇒ NamedNodeMap которая представляет собой информацию об атрибутах
- childElementCount ⇒ целое число, количество дочерних элементов узла Node
- children ⇒ HTMLCollection состоящая из дочерних элементов
- classList ⇒ DOMTokenList предоставляет удобный интерфейс для работы с элементами class (readonly)
- className ⇒ строка с классом (classList - для более удобной работы )
- clientHeight ⇒ height + padding для элементов без стилей, строчных == 0
- clientLeft ⇒ ширина от левого края в пикселях
- clientTop ⇒ Толщина верхней границы элемента в пикселях (без margin и padding)
- clientWidth ⇒ ширина элемента в пикселях
- currentCSSZoom ⇒ CSSZoom значение
- elementTiming (Экспериментальная возможность) - дял измерения производительности
- firstElementChild ⇒ Element первый дочерний
- id ⇒ строку с id
- innerHTML - устанавливает или получает внутреннюю разметку дочерних элементов
- lastElementChild ⇒ Element последний дочерний
- localName - локальное название узла
- namespaceURI ⇒ пространство имен
- nextElementSibling ⇒ Element последний элемент перед текущем
- outerHTML ⇒ включает в себя сам элемент и потомков

```js
//есть нюанс
var p = document.getElementsByTagName("p")[0];
console.log(p.nodeName); // показывает: "P"
p.outerHTML = "<div>Этот div заменил параграф.</div>";
console.log(p.nodeName); // всё ещё "P";
```

- part - ::part pseudo-element
- prefix - префикс NS
- previousElementSibling ⇒ Element элемент перед
- role - ариа роль
- scrollHeight - высота элемента с прокруткой
- scrollLeft - (getter/setter) на сколько прокрутить или прокручен элемент
- scrollLeftMax (Не стандартно) - мак кол-во на сколько можно прокрутить
- scrollTop (getter/setter) - на сколько прокрутить или прокручен
- scrollTopMax (Не стандартно) - мак кол-во на сколько можно прокрутить
- scrollWidth (readonly) - сколько невидно
- shadowRoot - Element.createShadowRoot() создаст shadow элемент
- slot - имя тенового слота
- tagName ⇒ имя элемента

## методы экземпляра

- after(node1, node2, nodeN) - вставляет после Element
- animate(keyframes, options) - создает и запускает анимацию
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
- getBoundingClientRect() ⇒ размер элемента и его расположение left, top, right, bottom, x, y, width и height
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

## события

- afterscriptexecute (Нестандартный Устаревший) - загрузка скрипта
- animationcancel (-ch,-ed) - незапланированное прерывание анимации
- animationend - окончание анимации
- animationiteration - окончание итерации анимации
- animationstart - начало анимации
- - свойства события:
- - - animationName
- - - elapsedTime
- - - pseudoElement
- auxclick - при клике не на основной мыши
- - свойства:
- - - altitudeAngle
- - - azimuthAngle
- - - pointerId
- - - width
- - - height
- - - pressure
- - - tangentialPressure
- - - tiltX
- - - tiltY
- - - twist
- - - pointerType
- - - .isPrimary
- beforeinput - при изменении в поле input, до самих изменений в dom элементе(не работаете на select)
- - свойства:
- - - data
- - - dataTransfer
- - - inputType
- - - isComposing
- beforematch (-ff, -sf)- для работы с hidden="until-found" срабатывает до того как найдет
- beforescriptexecute (Нестандартный Устаревший)
- beforexrselect (Экспериментальный) - WebXR
- blur - при потери фокуса с элемента (не всплывает)
- click - this - элемент на котором было вызвано = mousedown + mouseup
- compositionend - для отмены ввода в системе написания текста
- compositionstart
- compositionupdate
- contentvisibilityautostatechange - content-visibility: auto
- contextmenu - пкм
- copy - Clipboard API событие
- cut
- dblclick
- - свойства:
- - - altKey
- - - button
- - - buttons
- - - clientX
- - - clientY
- - - ctrlKey
- - - layerX
- - - layerY
- - - metaKey
- - - movementX
- - - movementY
- - - offsetX
- - - offsetY
- - - pageX
- - - pageY
- - - relatedTarget
- - - screenX
- - - screenY
- - - shiftKey
- - - mozInputSource
- - - webkitForce
- - - x === MouseEvent.clientX.
- - - y === MouseEvent.clientY.
- DOMActivate Устаревший
- DOMMouseScroll (Нестандартный Устаревший)
- focus - при фокусировке (не всплывает)
- focusin - при фокусировке (всплывает)
- focusout - потеря фокуса (всплывает)
- fullscreenchange - при переходе в fullscreen режим
- fullscreenerror
- gesturechange Нестандартный - при передвижении цифр во время касания
- gestureend Нестандартный
- gesturestart Нестандартный
- gotpointercapture - срабатывает если setPointerCapture()
- input - ввод
- keydown - нажата клавиша
- keypress - Устаревший - клавиши alt, enter
- keyup - клавиша отжата
- lostpointercapture

- mousedown - нажата кнопка мыши
- mouseenter - находится над элементом
- mouseleave - уходит с элемента
- mousemove - мышь внутри элемента
- mouseout - движения не внутри элеме6нта
- mouseover

mouseup
mousewheel (Нестандартный Устаревший)
MozMousePixelScroll (Нестандартный Устаревший)
paste
pointercancel
pointerdown
pointerenter
pointerleave
pointermove
pointerout
pointerover
pointerrawupdat e(Экспериментальный)
pointerup
scroll
scrollend
scrollsnapchang e(Экспериментальный)
scrollsnapchangin g(Экспериментальный)
securitypolicyviolation
touchcancel
touchend
touchmove
touchstart
transitioncancel
transitionend
transitionrun
transitionstart
webkitmouseforcechangedНестандартный
webkitmouseforcedownНестандартный
webkitmouseforceupНестандартный
webkitmouseforcewillbeginНестандартный
wheel

⇒
