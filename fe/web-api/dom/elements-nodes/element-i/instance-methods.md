методы экземпляра

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

# Методы вставки append, prepend, before, after, replaceWith

```html
<style>
  .alert {
  }
</style>

<script>
  //создаст div элемент со стилем div и текстом
  let div = document.createElement("div");
  div.className = "alert";
  div.innerHTML = "<strong>….";

  document.body.append(div);
</script>
```

- node.append(content) – добавляет узлы или строки в конец node
- node.prepend(content) – вставляет узлы или строки в начало node, prepend(param1, param2, paramN)
- node.before(content) – вставляет узлы или строки до node, before(param1, param2, paramN) - добавляет в начало Element
- node.after(content) – вставляет узлы или строки после node, after(node1, node2, nodeN)
- node.replaceWith(content) – заменяет node заданными узлами или строками, replaceWith(param1, param2, paramN) - заменит детей Element

в качестве контента могут быть строки или элементы

```html
<ol id="ol">
  <li>0</li>
  <li>1</li>
  <li>2</li>
</ol>

<script>
ol.before("before"); //вставить строку before перед ol
ol.after("after"); //вставить строку after после ol

let liFirst = document.createElement("li"); //создать элемент li присвоить переменной liFirst
liFirst.innerHTML = "prepend"; //содержимое liFirst строке prepend
ol.prepend(liFirst); //подставить перед ol

let liLast = document.createElement("li");
liLast.innerHTML = "append";
ol.append(liLast);
</script>

<!-- Методы могут вставлять несколько узлов -->
<div id="div"></div>
<script>
  div.before("<p>Привет<p>", document.createElement("hr")) //строчка привет с подчеркиванием;
<script>

```

```html
<!-- # задача дерево из объекта -->
<!-- Задача превратить объект в документ Вариант 1 с помощью DOM -->
<body>
    
  <div id="container"></div>
      
</body>
<script>
  "use strict";
  let data = {
    Рыбы: {
      форель: {},
      лосось: {},
    },
    Деревья: {
      Огромные: {
        секвойя: {},
        дуб: {},
      },
      Цветковые: {
        яблоня: {},
        магнолия: {},
      },
    },
  };
  function createTree(container, obj) {
    container.append(createTreeDom(obj));
  }

  function createTreeDom(obj) {
    if (!Object.keys(obj).length) return;
    //возвращать undefined при пустом объекте
    let ul = document.createElement("ul");
    //создаем Ul-элемент
    for (let key in obj) {
      let li = document.createElement("li"); //создаем li
      li.innerHTML = key;
      //внутрь li упаковываем key из цикла
      let childrenUL = createTreeDom(obj[key]); //рекурсивный вызов функции, если есть вложенные объекты
      if (childrenUL) {
        //при последнем шаге
        li.append(childrenUL); //добавить
      }
      ul.append(li);
    }
    return ul;
  }
  let container = document.getElementById("container");
  createTree(container, data);
</script>
```

Вариант 2 с помощью строк

```js
function createTree(container, obj) {
  container.innerHTML = createTreeText(obj);
  //упаковать в контейнер результат функции createTree
}
function createTreeText(obj) {
  let li = ""; //пустая строку в переменной li
  let ul; //объявим перченную ul
  for (let key in obj) {
    //для каждого свойства объекта
    li += "<li>" + key + createTreeText(obj[key]) + "</li>";
    // присвоить li значение ключа и рекурсивно вызвать функцию, li ожидает до последнего шага вложенности и не присваивается
  }
  if (li) {
    ul = "<ul>" + li + "</ul>"; //дошли до последнего шага вложенности, произошло присвоение переменной li, и все вложенные li оборачиваются в ul
  }
  return ul || ""; //функция возвращает ul или пустую строку на итерациях с пустым объектом
}
createTree(container, data);
```

<!--  -->

- attachShadow() - добавляет теневое DOM дерево к Element
- checkVisibility(options) ⇒ boolean виден ли элемент (display:none)
- - options:
- - - contentVisibilityAuto
- - - opacityProperty
- - - visibilityProperty
- - - checkOpacity
- - - checkVisibilityCSS

# closest

closest(selectors) ⇒ возвращает ближайший родительский компонент по селектору. elem.closest(css) ищет ближайшего предка, который соответствует css – селектору

```html
<h1>Содержание</h1>
<div class="contents">
  <ul class="book">
    <li class="chapter">Глава 1</li>
     
    <li class="chapter">Глава 2</li>
  </ul>
</div>

<script>
  let chapter = document.querySelector(".chapter");
  alert(chapter.closest(".book")); //UL
  alert(chapter.closest(".contents")); //DIV
  alert(chapter.closest("h1")); //null так как div не предок
</script>
```

<!--  -->

- computedStyleMap() ⇒ StylePropertyMapReadOnly
- getAnimations() ⇒ Animation
- getAttribute(attributeName) ⇒ значение атрибута по имени
- getAttributeNames() ⇒ массив с именами атрибутов
- getAttributeNode() ⇒ Attr
- getAttributeNodeNS() ⇒ Attr для NS
- getAttributeNS() ⇒ Attr для NS

<!--  -->

# getBoundingClientRect()

⇒ размер элемента и его расположение left, top, right, bottom, x, y, width и height. Возвращает координаты в контексте окна для минимального по размеру прямоугольника, который заключает в себе элемент elem в виде объекта встроенного класса DOMRect

x/y – координаты начла прямоугольника относительно окна
width/height – ширина высота прямоугольника
top/bottom – Y верхней нижней границы прямоугольника bottom = y + height
left/right – X координата левой правой границы прямоугольника right = y + height

в момент когда элемент уходит за область прокрутки, то возвращаются отрицательные координаты

<!--  -->

- getClientRects() ⇒ DOMRect[]

# getElementsByClassName

⇒ HTMLCollection из потомков с names в качестве classNames

<!--  -->

- getElementsByTagName() ⇒ HTMLCollection из потомков по имя тега
- getElementsByTagNameNS() ⇒ HTMLCollection из потомков по имя тега дял NS
- getHTML(options) ⇒ элементы DOM в виде строки
- hasAttribute(name) ⇒ boolean если ли данный атрибут на Element
- hasAttributeNS()
- hasAttributes() ⇒ boolean если ли атрибуты
- hasPointerCapture(pointerId)

# Методы вставки insertAdjacentHTML/text/Element

разновидности:

- insertAdjacentElement(position, element) - добавит Element в dom дерево. Чтобы вставить HTML как HTML
- insertAdjacentHTML(position, text) - вставит text как html
- insertAdjacentText(position, element)- вставит text как html

elem.insertAdjacentHTML(position, html)

position принимает значения:

- beforebegin – вставить html перед elem
- afterbegin – вставить html в начало
- beforend – вставить html в конец elem
- afterend – вставить html после elem

```html
<div id="div"></div>
<script>
  div.insertAdjacentHTML("beforebegin", "<p>Привет</p>");
  div.insertAdjacentHTML("afterbegin", "<p>Пока<p>");
<script>
```

```html
<style>
  .alert {
  }
</style>
<script>
  document.insertAdjacentHTML(
    "afterbegin",
    '<div class="alert"><strong>Всем</strong>Вы прочитали важное сообщение</div>',
  );
</script>
```

<!--  -->

# matches

matches(selectorString) - соответствует ли Element css селектору selectorString. Ничего не ищет в проверяет удовлетворяет ли elem CSS – селектору и возвращает true или false. Удобно для перебора элементов массива

```html
<a href="http:/example.com/file.zip"></a>
<a href="http:/ya.ru"></a>
</body>
<script>
for(let elem of document.body.children) {
  if(elem.matches('a[href$="zip"]')){
    alert("Ссылка на архив: " + elem.href);
    }
  }
</script>
```

<!--  -->

- moveBefore(movedNode, referenceNode) - movedNode переместит перед referenceNode

# querySelector

querySelector(selectors) ⇒ первый Element по selectors

```js
elem.querySelector("css-rule"); //возвращает первый элемент соответствующий CSS-селектору
elem.querySelectorAll("css-rule")[0] == elem.querySelector("css-rule");
```

```html
<!-- живые коллекции -->
<div>Оба тега DIV внизу невидимы</div>

<div hidden>С атрибутом hidden</div>
<div id="elem">с назначенным JS свойством "hidden"</div>

<script>
  elem.hidden = true;
</script>

<!-- Мигающий элемент -->

<div id="elem">Мигающий элемент</div>

<script>
  setInterval(() => (elem.hidden = !elem.hidden), 1000);
</script>
```

- !!!querySelector возвращают статическую коллекцию

# querySelectorAll

querySelectorAll(selectors) ⇒ статичный NodeList. Возвращает все элементы внутри elem, удовлетворяющий CSS – селектору

```html
<ul>
   
  <li>Этот</li>
     
  <li>Текст</li>
</ul>
<ul>
   
  <li>полностью</li>
     
  <li>пройден</li>
</ul>
 
<script>
  // Запрос получает все элементы li которые являются потомками в ul
  let elements = document.querySelectorAll("ul > li:last-child"); //все потомки, которые являются последними потомками в <ul>
  for (let elem of elements) {
    alert(elem.innerHTML); //тест, пройден
  }
</script>
```

Распространяется также на псевдо-классы

<!--  -->

- releasePointerCapture() - остановит Pointer Capture

# remove()

удалит элемент из dom

```html
<style>
.alert {ы}
</style>
<script>
  let div = document.createElement("div");
  div.className = "alert";
  div.innerHTML = "Строка"

  document.body.append("div");
  setTimeout(()=> div.remove(), 1000);
</script>

<!-- Все методы вставки удаляют узлы со старых мест, если нужно переместить, то не нужно его удалять -->
<div id="first">Первый</div>
<div id="second">Второй</div>
<script>
  second.after(first); // after, before, prepend, append
<script>
```

<!--  -->

- removeAttribute(attrName) - удалит атрибут с Element
- removeAttributeNode() - удалит Attr с Element
- removeAttributeNS()
- replaceChildren(param1, param2, paramN) - заменит Element элементами param1, param2, paramN
- requestFullscreen() - сделать Element на весь экран
- requestPointerLock() - блокировка курсора

# scroll

Варианты использования:

- scroll(xCoord, yCoord) - прокрутить до (xCoord, yCoord)
- scroll(options) - прокрутить с учетом behavior
- - options:
- - - behavior: smooth, instant, auto

# scrollBy()

scrollBy(xCoord, yCoord) или scrollBy(options)

# scrollIntoView()

прокрутить до элемента

```ts
elem.scrollIntoView({
  // настройки анимации
  behavior: "auto", // "smooth" | "instant",
  // выравнивание по вертикальной оси элемента
  block: "start", // "center" | "end" | "nearest",
  container: "all", // "nearest",
  // // выравнивание по горизонтально оси элемента
  inline: "nearest", // "start" | "center" | "end",
});
```

scrollIntoView(alignToTop) - элемент в поле видимости alignToTop - будет ли верхняя граница по верху

# scrollIntoViewIfNeeded()

Не стандартно

# scrollTo()

```js
elem.scrollTo(xCoord, yCoord);
```

прокрутка внутри данного элемента

<!--  -->

- setAttribute(name, value) - установить атрибут name значение value
- setAttributeNode(attribute) - attribute - Attr
- setAttributeNodeNS()
- setAttributeNS()
- setCapture() - Не стандартноУстарело
- setHTMLUnsafe() - для преобразования строки в HTML
- setPointerCapture() - сделать элемент target для Pointer Events
- toggleAttribute(name, force) - для boolean атрибутов
