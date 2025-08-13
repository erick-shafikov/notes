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

# dom дочерние childNodes, firstChild, lastChild

childNodes похож на массив, но это просто перебираемый объект, для перебора мы используем for…of

- children – коллекция детей, которые являются элементами
- firstElementChild, LastElementChild – первый и последний дочерние элементы
- parentElement – родитель-элемент, тоже самое что и parentNode, исключение document.documentElement
- !!! childNodes, firstChild, lastChild - могут вернуть пробелы, переносы строк, лучше использовать element-поисковики

```js
for (let node of document.body.childNodes) {
  //не работают методы массивов
  alert(node);
}

// Превратим в массив
alert(Array.from(document.body.childNodes).filter);
```

!!!Все коллекции только для чтения и отражают текущее состояние DOM
!!!Лучше не использовать цикл for…in
!!!Только для чтения

# соседи и родитель nextSibling, previousSibling

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

проход по всем родителям

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
//[object Text]
//[object HTMLDivElement]
//[object Text]</div>
//[object HTMLUlListElement]
//[object Text] </li>
//[object Text] Конец //[object Text] </div>
//[object HTMLScriptElement]
//childNodes не массив, а перебираемый объект
//text, DIV, Text, UL, … SCRIPT
</script>

```

- !!!Все коллекции только для чтения и отражают текущее состояние DOM
- !!!Лучше не использовать цикл for…in
- !!!Только для чтения

# навигация по элементам

- children – коллекция детей, которые являются элементами
- firstElementChild, LastElementChild – первый и последний дочерние элементы
- previousElementSibling, nextElementSibling – соседи элементы
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

# Навигация в таблицах

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
