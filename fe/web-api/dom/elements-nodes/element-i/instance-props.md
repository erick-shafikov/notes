свойства экземпляра

# aria-свойства

- aria-свойства: ariaActiveDescendantElement, ariaAtomic, ariaAutoComplete, ariaBrailleLabel, ariaBrailleRoleDescription, ariaBusy, ariaChecked, ariaColCount, ariaColIndex, ariaColIndexText, ariaColSpan, ariaControlsElements, ariaCurrent, ariaDescribedByElements, ariaDescription, ariaDetailsElements, ariaDisabled, ariaErrorMessageElements, ariaExpanded, ariaFlowToElements, ariaHasPopup, ariaHidden, ariaInvalid, ariaKeyShortcuts, ariaLabel, ariaLabelledByElements, ariaLevel, ariaLive, ariaModal, ariaMultiLine, ariaMultiSelectable, ariaOrientation, ariaOwnsElements, ariaPlaceholder, ariaPosInSet, ariaPressed, ariaReadOnly, ariaRelevantНе стандартно, ariaRequired, ariaRoleDescription, ariaRowCount, ariaRowIndex, ariaRowIndexText, ariaRowSpan, ariaSelected, ariaSetSize, ariaSort, ariaValueMax, ariaValueMin, ariaValueNow, ariaValueText

# assignedSlot

⇒ HTMLSlotElement, если есть shadow-root элементы

# attributes

⇒ NamedNodeMap которая представляет собой информацию об атрибутах, класс Attr наследует Node

```js
// получить
attr = element.attributes;
```

свойства экземпляра:

- name - Имя атрибута
- namespaceURI
- localName
- prefix
- ownerElement ⇒ элемент
- value

```html
<label test="initial value"></label>

<button>Click me to set test to <code>"a new value"</code>…</button>

<p>
  Current value of the <code>test</code> attribute:
  <output id="result">None.</output>
</p>
```

```js
const element = document.querySelector("label");
const button = document.querySelector("button");
const result = document.querySelector("#result");

const attribute = element.attributes[0];
result.value = attribute.value;

button.addEventListener("click", () => {
  attribute.value = "a new value";
  result.value = attribute.value;
});
```

Когда у элемента есть id или другой стандартный атрибут создается соответствующее свойство, но если атрибут нестандартный, то этого не происходит. HTML атрибуты регистронезависимые значения – строки

```html
<body id="test" something="non-standard">
  <script>
    alert(document.body.id); //test
    alert(document.body.something); //undefined
    // методы работы с атрибутами
    elem.hasAttribute(name); //проверяет наличие атрибута
    elem.getAttribute(name); //получает значение атрибута
    elem.setAttribute(name, value); //устанавливает значение атрибута
    elem.removeAttribute(name); //удаляет атрибут
    elem.attributes; //коллекция объектов
  </script>
</body>
```

Синхронизация между атрибутами и свойствами

```html
<input />

<script>
  let input = document.querySelector("input");
  input.setAttribute("id", "id");
  alert(input.id); //id
  input.id = "newId";
  alert(input.getAttribute("id")); //newId
</script>

<!-- Исключение для input.value -->

<input />

<script>
  let input = document.querySelector("input");
  input.setAttribute("value", "text");
  alert(input.value); //text

  input.value = "newValue";
  alert(input.getAttribute("Value")); //text не обновилось
</script>
```

DOM свойства типизированы, DOM свойства не всегда являются строками

```html
<input id="input" type="checkbox" checked />
<!--checkbox-->

<script>
  alert(input.getAttribute("checked")); //значение пустая строка
  alert(input.checked); //значение свойства true
</script>

<!-- свойство style является объектом -->

<div id="div" style="color:red; font-size:120">Hello</div>
<script>
  alert(div.getAttribute("style")); //{color: red; font-size: 120%}

  alert(div.style); //[object CSSStyleDeclaration]
  alert(div.style.color); //red
</script>

<!-- Свойство href всегда содержит полный URL, даже если содержит относительный путь
или # -->

<a id="a" href="#hello">link</a>

<script>
  alert(a.getAttribute("href")); //#hello
  alert(a.href); //полный URL
</script>
```

# Нестандартные атрибуты

```html
<!-- Заполнить html соответствующими элементами помечать div чтобы показать что здесь поле name -->
<div show-info="name"></div>
<!-- здесь age -->
<div show-info="age"></div>

<script>
  let user = {
    name: "Pete",
    age: 25
  };

  for(let div of document.querySelectorAll["show-info"]) {
   let field = div.getAttribute("show-info");
   div.innerHTML = user[field];
  }
<script>

```

```html
<style>
  .order[order-state="new"] {
    color: green;
  }

  .order[order-state="pending"] {
    color: blue;
  }

  .order[order-state="canceled"] {
    color: red;
  }
</style>

<div class="order" order-state="new">A new order</div>
<div class="order" order-state="pending">A pending-order</div>
<div class="order" order-state="canceled">A canceled order</div>
```

все атрибуты data- зарезервированы для использования программистами

если у elem есть атрибут data-about то обратиться к нему можно как elem.dataset.about

```html
<body data-about="Elephants">
<script>
  alert(document.body.dataset.about); //Elephants
<script>

```

атрибуты состоящие из нескольких слов data-order-state становится свойствами dataset.orderState

```html
<style>
.order[order-state="new"] {color: green;}
.order[order-state="pending"] {color: blue;}
.order[order-state="canceled"] {color: red;}
</style>

<body>
  <div class="order" data-order-state="new">A new order</div>
  <div class="order" data-order-state="pending">A pending-order</div>
  <div class="order" data-order-state="canceled">A canceled order </div>
</body>

<script>
  alert(order.dataset.orderState); //new
<script>

```

# childElementCount

⇒ целое число, количество дочерних элементов узла Node

# children

⇒ HTMLCollection состоящая из дочерних элементов

<!--  -->

# classList

⇒ DOMTokenList предоставляет удобный интерфейс для работы с элементами class (readonly). Предоставляет основной метод работы с классами

```js
elem.classList.add("class"); //– добавить класс
elem.classList.remove("class"); //- удаляет класс
elem.classLIst.toggle("class"); //– добавить класс, если его нет иначе удалить
elem.classList.contains("class"); //- проверка на наличие
```

```html
<body class="main page">
   
  <script>
    document.body.classList.add("article");
    alert(document.body.className); //main page article

    for (let name of document.body.classList) {
      alert(name); //name, page, article
    }
  </script>
</body>
```

# className

⇒ строка с классом (classList - для более удобной работы )

класс как свойство className: elem.className соответствует атрибуту класс

```html
<body class="main page">
   
  <script>
    alert(document.body.className); //main page
  </script>
</body>
```

!!!При присваивании заменяет полностью строку класса

# clientHight, clientWidth (содержимое + padding)

Что бы получить размеры окна можно взять свойства document.Element.clientHeight, document.Element.clientWidth. Размер области внутри рамок элемента, если нет внутренних отступов padding то clientWidth/height в точности равны размеру содержимого, clientHeight ⇒ height + padding для элементов без стилей, строчных == 0

отличие между CSS св-ва width и client Width:

clientWidth возвращает число, а getComputedStyle(elem).width – строку с px на конце. getComputedStyle не всегда даст ширину, он может вернуть, к примеру, "auto" для строчного элемента. clientWidth соответствует внутренней области элемента, включая внутренние отступы padding, а CSS-ширина (при стандартном значении box-sizing) соответствует внутренней области без внутренних отступов padding. Если есть полоса прокрутки, и для неё зарезервировано место, то некоторые браузеры вычитают его из CSS-ширины (т.к. оно больше недоступно для содержимого), а некоторые – нет. Свойство clientWidth всегда ведёт себя одинаково: оно всегда обозначает размер за вычетом прокрутки, т.е. реально доступный для содержимого

в отличие от window.innerHight и window.innerWidth – указывают на видимую часть документа , то есть и на часть с прокруткой
из-за отличий в браузерах следует брать максимальное значение

```js
let scrollHeight = Math.max(
  document.body.scrollHeight,
  document.documentElement.scrollHeight,
  document.body.offsetHeight,
  document.documentElement.offsetHeight,
  document.body.clientHeight,
  document.documentElement.clientHeight,
);
```

# clientTop, clientLeft (border)

для метрик border выражает ширину рамки, но на самом деле это отступы внутренней части элемента от внешней

<!--  -->

- currentCSSZoom ⇒ CSSZoom значение
- elementTiming (Экспериментальная возможность) - дял измерения производительности
- firstElementChild ⇒ Element первый дочерний
- id ⇒ строку с id

# innerHtml

устанавливает или получает внутреннюю разметку дочерних элементов, позволяет получить HTML – содержимое элемента в виде строки, так же выступает сеттером

```html
 
<body>
     
  <p>Параграф</p>
     
  <div>DIV</div>
   
  <script>
    alert(document.body.innerHTML); //читаем текущее содержимое
    document.body.innerHTML = "Новый BODY!"; //меняем содержимое
  </script>
</body>
<!-- При вставке некорректного HTML браузер исправит ошибки -->
<body>
   
  <script>
    document.body.innerHTML = "<b>тест"; //забыли закрыть тег
    alert(document.body.innerHTML); //<b>тест</b> исправлено
  </script>
</body>
```

!!!но если вставить тег script - он становится частью HTML но не запускается
!!!innerHTML += осуществляет перезапись, старое содержимое удаляется на его место встает новая запись

```js
// Сортировка таблицы
let sorted = Array.from(table.rows) //ряды таблицы в массив
  .slice(1) //первая строка не нужна, начать сортировку со второй
  .sort((rowA, rowB) =>
    rowA.cells[0].innerHTML > rowB.cells[0].innerHTML ? 1 : -1,
  ); //сортируем таблицу , где cells внутри tr
table.tBodies[0].append(...sorted);
```

<!--  -->

- lastElementChild ⇒ Element последний дочерний
- localName - локальное название узла
- namespaceURI ⇒ пространство имен
- nextElementSibling ⇒ Element последний элемент перед текущем

# outerHTML

⇒ включает в себя сам элемент и потомков. Свойство outerHTML содержит HTML элемента целиком. Это как innerHTML плюс сам элемент

```html
<div id="elem">Привет<b>Мир</b></div>

<script>
  alert(elem.outerHTML); //<div id="elem">Привет<b>Мир</b></div>
</script>
<!-- в отличие от innerHTML запись в outerHTML не изменяет элемент, вместо этого
элемент заменяется во внешнем контексте -->
<div>Привет, мир!</div>

<script>
  let div = document.querySelector("div");
  div.outerHTML = "<p>Новый элемент</p>";
  // 1. div был удален из документа
  //2. Вместо него вставлен другой HTML
  //3. в div осталось старое значение
  alert(div.outerHTML); //<div>Привет, мир!</div>
</script>
```

```js
//есть нюанс
var p = document.getElementsByTagName("p")[0];
console.log(p.nodeName); // показывает: "P"
p.outerHTML = "<div>Этот div заменил параграф.</div>";
console.log(p.nodeName); // всё ещё "P";
```

<!--  -->

- part - ::part pseudo-element
- prefix - префикс NS
- previousElementSibling ⇒ Element элемент перед
- role - ариа роль

<!--  -->

# scrollWidth, scrollHeight

scrollWidth, scrollHeight - свойства как clientWidth и clientHeight включающие в себя прокрученную область, которую не видно

# scrollLeft, scrollTop

прокрученной в данный момент части элемента, ScrollTop – то то, сколько прокручено сверху scrollTop на 0 или на infinity – прокручено вверх или до конца

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

<!--  -->

- scrollLeftMax (Не стандартно) - мак кол-во на сколько можно прокрутить
- scrollTopMax (Не стандартно) - мак кол-во на сколько можно прокрутить
- shadowRoot - Element.createShadowRoot() создаст shadow элемент
- slot - имя тенового слота
- tagName ⇒ имя элемента, есть только у элементов Element
