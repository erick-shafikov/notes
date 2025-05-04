# навигация по элементам

В операции начинаются с объекта document из него мы можем получить доступ к любому элементу
Сверху – documentElement и body
Самый верхний узел документа доступен как свойства объекта document

```html
<html>
  <!-- html === document.documentElement, в DOM он соответствуй тегу -->
  <html>
    <body>
      <!--body ===  document.body -->
      <head>
        <!-- head === document.head -->
      </head>
    </body>
  </html>
</html>
```

document.body может быть равен null если скрипт находится в head, document.body в нем недоступен html. В DOM null означает - не существует

```html
<html>
<head>
  <script>
    alert("Из HEAD" + document.body); //null
  </script>
</head>
<body>
  <script>
    alert("ИЗ BODY" + document.body); // HTMLBodyElement, теперь он есть
  </script>
<body>
</html>

```

## childNodes, firstChild, lastChild

childNodes похож на массив, но это просто перебираемый объект, для перебора мы используем for…of

- children – коллекция детей, которые являются элементами
- firstElementChild, LastElementChild – первый и последний дочерние элементы
- parentElement – родитель-элемент, тоже самое что и parentNode, исключение document.documentElement

```js
document.documentElement.parentNode; //document
document.documentElement.parentElement; //null, т.к. document – не элемент

while ((elem = elem.parentElement)) {
  //идти наверх до <html>
  alert(elem);
}
```

```html
<body>
  <!-- [object HTMLDivElement] -->
  <div> Начало </div>
<ul>
  <li> Информация </li>
  <!-- [object HTMLUlListElement] -->
<ul>
<!-- [object HTMLDivElement] -->
<div> Конец </div>
</body>
<script>
for (elem of document.body.children) {
  alert(elem); //DIV, UL, DIV, SCRIPT
}
</script>

```

```html
<body>
  <div> Начало </div>
  <ul>
    <li>
      <b> Информация </b>
      </li>
      <ul>
</body>


<!-- childNodes содержит список всех детей, включая текстовые узлы
//[object Text]
//[object HTMLDivElement]
//[object Text]</div>
//[object HTMLUlListElement]
//[object Text] </li>
<//[object Text] Конец //[object Text] </div>
//[object HTMLScriptElement]
//childNodes не массив, а перебираемый объект
//text, DIV, Text, UL, … SCRIPT
-->
```

```js
for (let node of document.body.childNodes) {
  //не работают методы массивов
  alert(node);
}

// Превратим в массив
alert(Array.from(document.body.childNodes).filter);
```

- !!!Все коллекции только для чтения и отражают текущее состояние DOM
- !!!Лучше не использовать цикл for…in
- !!!Только для чтения

<!-- соседи и родитель nextSibling, previousSibling ------------------------------------------------------------------------------------------>

## соседи nextSibling, previousSibling

Соседи – это объекты у которых один родитель

- previousElementSibling, nextElementSibling – соседи элементы

```html
<html>
  <head>
    …
  </head>
  <body>
    …
  </body>
  <!-- head и body соседи - правый и левый -->
</html>
```

```js
alert(document.body.parentNode === document.documentElement); //true <html> является родителем <body>
// .nextSibling – следующий сосед
alert(document.head.nextSibling); //HTMLBodyElement <body> идет после <head>
// .previousSibling – предыдущий сосед
alert(document.body.previousSibling); //HTMLHeadElement
```

<!-- Навигация в таблицах -------------------------------------------------------------------------------------------------------------------->

## Навигация в таблицах

- table.rows – коллекция строк tr таблицы
- table.caption/tHead/tFoot – ссылки на элементы таблицы caption thead tfoot
- table.tBodies – коллекция элементов таблицы tbody

thead, tfood, tbody - представляют свойство rows
tbody.rows – коллекция строки таблицы

tr Тег служит контейнером для создания строки таблицы. Каждая ячейка в пределах такой строки может задаваться с помощью тега th или td. Синтаксис:

```html
<table>
  <tr>
    <td>...</td>
  </tr>
</table>
```

- tr.cells – коллекция td и th ячеек находящихся внутри строки tr
- tr.sectionRowIndex – номер строки tr в текущей секции thead/tbody/tfoot
- tr.rowIndex номер строки tr в таблице (включая все строки столбцы)

- th тег предназначен для создания одной ячейки таблицы, которая обозначается как заголовочная. Текст в такой ячейке отображается браузером обычно жирным шрифтом и выравнивается по центру. Тег th должен размещаться внутри контейнера tr, который в свою очередь располагается внутри тега table.

- td тег Предназначен для создания одной ячейки таблицы. Тег td должен размещаться внутри контейнера tr, который в свою очередь располагается внутри тега table.

- td.cellIndex – номер ячейки в строке <tr>

```html
<table id="" table="">
  <tr>
    <td>один</td>
    <td>два</td>
  </tr>
  <td>три</td>
  <td>четыре</td>

  <script>
    alert(table.rows[0].ceil[1].innerHTML); //"два"
  </script>
</table>
```

# поиск элементов в документе

<!-- getElementById ------------------------------------------------------------------------------------------------------------------------>

## getElementById

document.getElementById(id) или просто id Если у элемента есть атрибут id, то его можно получить где бы он не находился

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

<!-- getElementsByTagName ------------------------------------------------------------------------------------------------------------------>

## getElementsByTagName

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

!!!Возвращает коллекцию а не элемент
!!!Коллекции отображают текущее состояние DOM

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

<!-- getElementsByClassName ---------------------------------------------------------------------------------------------------------------->

## getElementsByClassName

возвращает все элементы, которые имеют данный CSS-класс

<!-- querySelectorAll ---------------------------------------------------------------------------------------------------------------------->

## querySelectorAll

- elem.querySelectorAll(css) возвращает все элементы внутри elem, удовлетворяющий CSS – селектору

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

<!-- querySelector ------------------------------------------------------------------------------------------------------------------------->

```js
elem.querySelector("css-rule"); //возвращает первый элемент соответствующий CSS-селектору
elem.querySelectorAll("css-rule")[0] == elem.querySelector("css-rule");
```

## живые коллекции

```html
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

<!-- matches ------------------------------------------------------------------------------------------------------------------------------->

## matches

elem.matches(css) – ничего не ищет в проверяет удовлетворяет ли elem CSS – селектору и возвращает true или false. Удобно для перебора элементов массива

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

<!-- closest ------------------------------------------------------------------------------------------------------------------------------->

## closest

elem.closest(css) ищет ближайшего предка, который соответствует css – селектору

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

<!-- contains ------------------------------------------------------------------------------------------------------------------------------>

## contains

elemA.contains(elemB) вернет true если elemB находится внутри elemA
