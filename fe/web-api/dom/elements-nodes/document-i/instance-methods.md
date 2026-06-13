# методы экземпляра

- adoptNode() - позволяет вставить кусок из одного документа в другой

```js
const iframe = document.querySelector("iframe");
const iframeImages = iframe.contentDocument.querySelectorAll("img");
const newParent = document.getElementById("images");

iframeImages.forEach((imgEl) => {
  newParent.appendChild(document.adoptNode(imgEl));
});
```

- append(param1, param2, paramN) - вставит что-то в конец документа, где param1, param2, paramN - узлы для вставки

- browsingTopics() - темы (Experimental Non-standard)
- caretPositionFromPoint(x, y, options) ⇒ CaretPosition объект
- caretRangeFromPoint() - позиция курсора в тексте (Non-standard)
- clear() - Deprecated
- close() - закроет запись в документ

```js
// открытие документа для записи в него.
// запись содержимого документа.
// закрытие документа.
document.open();
document.write("<p>The one and only content.</p>");
document.close();
```

- createAttribute() - создает атрибут дял использования в тегах

```js
const node = document.getElementById("div1");
const a = document.createAttribute("my_attrib");
a.value = "newVal";
node.setAttributeNode(a);
console.log(node.getAttribute("my_attrib")); // "newVal"
```

- createAttributeNS() - createAttribute + NS
- createCDATASection() - создает CDATA узел
- createComment() - создаст коммент
- createDocumentFragment() - интерфейс для создания фрагментов

## createElement()

createElement(tagName, {is:'user-element-name'}) ⇒ элемент

```js
document.createElement("tag-name"); //- создает элемент с заданным тэгом
let div = document.createElement("div");

document.createTextNode(text); //– создает текстовый узел с заданным текстом
let textNode = document.createTextNode("А вот и я");

// Создать элемент с текстом
let div = document.createElement("div");
div.className = "alert";
div.innerHTML = "<strong>Всем привет</strong> Вы прочитали важное сообщение"; //создали элемент, но пока он не является частью документа
```

<!--  -->

- createElementNS()
- createEvent() - Deprecated
- createExpression() - дял XPathExpression
- createNodeIterator(root, whatToShow, filter)
- - root - куда встраивать
- - whatToShow - 0xFFFFFFFF по умолчанию
- - filter - функция
- createNSResolver() - Deprecated
- createProcessingInstruction() - для XML

- createRange(startNode, startOffset) ⇒ Range.
- createTextNode(data) - создаст текстовый узел
- - data - текстовый контент
- createTouch() - Non-standard Deprecated
- createTouchList() - Non-standard Deprecated
- createTreeWalker() ⇒ TreeWalker

<!--  -->

## elementFromPoint(x, y)

возвращает самый глубоко вложенный элементы окне находящийся по координатам (x,y)

<!--  -->

- elementsFromPoint() ⇒ Element[] от самого верхнего до самого нижнего
- enableStyleSheetsForSet() - Non-standard Deprecated
- evaluate() ⇒ XPathResult
- execCommand() - Deprecated для работы с документов в режиме редактирования

- exitFullscreen() - для выхода из полноэкранного режима
- exitPictureInPicture() - для выхода из PictureInPicture режима
- exitPointerLock() -
- getAnimations() ⇒ Animation[] со всеми анимаций
<!--  -->

## getElementById

⇒ Element по id, регистр зависимый

document.getElementById(id) или просто id Если у элемента есть атрибут id, то его можно получить где бы он не находился. При существовании двух элементов с одинаковым id - вернет первый

```html
<div id="elem-content">Element</div>
<!-- так как в названии есть дефис, мы можем к нему обратиться window["elem-content"]  -->
<!-- так как есть дефис такой id не может служить именем переменной -->
<!-- если в скрипте есть такая же переменная, то она перекрывает переменную в DOM -->

<script>
  let elem = document.getElementById("elem");
  elem.style.background = "red";
</script>
<!-- !!!Значение id должно быть уникальным -->

<div id="elem">
  <div id="elem-content">Элемент</div>
</div>
<script>
  elem.style.background = "red";
</script>
```

# getElementsByClassName

⇒ Element[] с указанным классом

# getElementsByName

⇒ Element[] с указанным атрибутом name

## getElementsByTagName

⇒ Element[] по тегу

ищет элементы с данным тегом и возвращает из коллекцию. Передав \* можно получить всех потомков

```html
<script>
  let divElements = document.getElementsByTagName("div");
  alert(divElements.length); //1
</script>

<div>Second div</div>

<script>
  alert(divElements.length); //2
</script>
<!-- !!!!querySelectorAll возвращает статическую коллекцию -->
<div>First div</div>

<script>
  let divElements = document.querySelectorAll("div");
  alert(divElements.length); //1
</script>

<div>Second div</div>

<script>
  alert(divElements.length); //1
</script>
```

- !!!Возвращает коллекцию а не элемент
- !!!Коллекции отображают текущее состояние DOM

```html
<body>
  <label> Младше 18 </label>
  <input type="radio" name="age" value="young" checked />Младше 18

  <label> от 18 до 50 </label>
  <input type="radio" name="age" value="mature" />от 18 до 50

  <label> старше 60 </label>
  <input type="radio" name="age" value="senior" /> старше 60
</body>

<script>
  let divElements = document.getElementByTagName("div"); // получить все div-элементы
  let inputs = table.getElementsByTagName("input");
  for (let input of inputs) {
    alert(input.value + ":" + input.checked); //young: true, mature: false, senior: false
  }
</script>
```

<!--  -->

- getElementsByTagNameNS() - с учетом NS
- getSelection() => Selection в котором содержится информация о выделенном тексте
- hasFocus() - имеет ли элемент или вложенные фокус
- hasStorageAccess()⇒Promise(Boolean) есть ли доступ к cookie
- hasUnpartitionedCookieAccess() ⇒ Promise(Boolean) есть ли доступ к сторонними cookie
- importNode() - создаст копию узла

```js
var iframe = document.querySelector("iframe");
var oldNode = iframe.contentWindow.document.getElementById("myNode");
var newNode = document.importNode(oldNode, true);
document.getElementById("container").appendChild(newNode);
```

- moveBefore(movedNode, referenceNode) - двигает movedNode, referenceNode - перед чем вставить
- mozSetImageElement(imageElementId, imageElement) - Non-standard - меняет фон
- open() - откроет возможность редактировать элемент

```js
//первое использование
document.open();
document.write("<p>Hello world!</p>");
document.write("<p>I am a fish</p>");
document.write("<p>The number is 42</p>");
document.close();

//второе открытие ссылок
document.open("https://www.github.com", "", "noopener=true");
```

- prepend(param1, param2, paramN) - вставит перед документом
- queryCommandEnabled() - Non-standard,Deprecated
- queryCommandState() - Non-standard, Deprecated
- queryCommandSupported() - Non-standard Deprecated
- querySelector(selectors) => первый Element в соответствии с селектором

# querySelectorAll

querySelectorAll() => нединамически NodeList

<!--  -->

- releaseCapture() - Non-standard
- replaceChildren(param1, param2) - для замены дочерних элементов
- requestStorageAccess({all:false, cookies, sessionStorage, localStorage, indexedDB}) - запрос для сторонних куки файлов
- requestStorageAccessFor(requestedOrigin) - Experimental - установит разрешение
  -startViewTransition(updateCallback) => ViewTransition для SPA контролировать переходы между страницами, updateCallback вызывается после перехода
- write() - Deprecated - открывает документ для редактирования, записывает html на страницу может быть сгенерирован по ходу
- writeln() - Выводит в документ строку со знаком перевода каретки в конце.
