# Document Object Model

Объектная модель HTML документа, то как браузер представляет HTML страницу в виде объекта, в DOM дереве существует 12, такие как document, тэги, текстовые узлы, комментарии, атрибуты. Все что есть в HTML есть в DOM дереве.

Каждый узел дерева – объекты. Теги являются узлами – элементами head и body его дочерние элементы

- !!!Пробел и перевод строки перед тегом head игнорируются
- !!!Если записываем что-то после тега body браузер автоматически перемещает эту запись в конец body, то есть после body не может быть никаких пробелов

Каждый DOM-элемент соответствует встроенному классу

- Event – корневой абстрактный класс
- Node – основной для DOM узлов, обеспечивает функциональность для parentNode, nextSibling, childNOdes (это все геттеры)
- Element – базовый класс для DOM обеспечивает навигацию

# оптимизация работы с DOM:

- В скриптах минимизируйте любую работу с DOM. Кэшируйте всё: свойства, объекты, если подразумевается повторное их использование. - При сложных манипуляциях разумно работать с «offline» элементом (т.е. который находится не в DOM, а в памяти), с последующим помещением его в DOM.
- Упрощать css – селекторы, чем меньше вложенность тем лучше
- Минимизировать любую работу DOM, изменять однократно
- Для изменений стилей лучше оперировать атрибутом class
- Анимировать лучше элементы позиционировать абсолютно или фиксировано, не изменять геометрию узлов
- откладывать невидимые изменения
- использовать web workers для больших и сложных задач
- использование requestAnimationFrame

# иерархия наследования (!!!TODO)

# Навигация по элементам (!!!TODO)

# размеры элемента (!!!TODO)

# стили элемента (!!!TODO)

# изменение контента

Методы изменения Node:

- [textContent](./elements-nodes/node-i.md#textcontent)
- [nodeValue/data](./elements-nodes/node-i.md#nodevaluedata)

Методы изменения Element:

- [innerHtml](./elements-nodes/element-i/instance-props.md#innerhtml)
- [outerHTML](./elements-nodes/element-i/instance-props.md#outerhtml)

Создание элементов:

- [createElement](./elements-nodes/document-i.md#createelement)

методы вставки:

- [appendChild, insertBefore, replaceChild, removeChild](./elements-nodes/node-i.md#appendchild-insertbefore-replacechild-removechild)
- [append, prepend, before, after, replaceWith](./elements-nodes/element-i/instance-methods.md#методы-вставки-append-prepend-before-after-replacewith)
- [Методы вставки insertAdjacentHTML/text/Elemen](./elements-nodes/element-i/instance-methods.md#методы-вставки-insertadjacenthtmltextelement)
- [удаление](./elements-nodes/element-i/instance-methods.md#remove)

методы клонирования:

- [cloneNode](./elements-nodes/node-i.md#clonenode)

# Работа с координатами элемента:

- [вычисление позиции мыши координаты от видимой области](./events/ui-events/mouse-event/event-props.md#clintx-clienty)
- [координаты от начала страницы](./events/ui-events/mouse-event/event-props.md#pagex-pagey)
- [координаты от начала экрана](./events/ui-events/mouse-event/event-props.md#screenx)

# работа со скроллом:

- [прокрутка до определенных координат страницы](./elements-nodes/element-i/instance-methods.md#scrolltox-coord-y-coord)
- [прокрутка на определенное количество пикселей](./elements-nodes/element-i/instance-methods.md#scroll)
- [прокрутка до определенного элемента](./elements-nodes/element-i/instance-methods.md#scrollintoview)

# Навигация:

- !!!Все коллекции только для чтения и отражают текущее состояние DOM
- !!!Лучше не использовать цикл for…in
- !!!Только для чтения

- [навигация в таблицах](./elements-nodes/html-elements/html-table-elements/table.md#навигация-в-таблицах)

## навигация по элементам

- [дочерние](./elements-nodes/element-i/instance-props.md#children)
- [дочерние](./elements-nodes/element-i/instance-props.md#firstelementchild-lastelementchild)
- [соседские](./elements-nodes/element-i/instance-props.md#previouselementsibling-nextelementsibling)

В операции начинаются с объекта document из него мы можем получить доступ к любому элементу. Сверху – documentElement и body.Самый верхний узел документа доступен как свойства объекта document

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

## навигация по узлам

- [дочерние](./elements-nodes/node-i.md#childnodes-firstchild-lastchild) - могут вернуть пробелы, переносы строк, лучше использовать element-поисковики
- [соседи и родитель](./elements-nodes/node-i.md#nextsibling-previoussibling)

# поиск элементов

- поиск по классу [getElementsByClassName в Document](./elements-nodes/document-i/instance-methods.md#getelementsbyclassname) и в [Element](./elements-nodes/element-i/instance-methods.md#getelementsbyclassname)
- поиск по селектору [querySelectorAll в Document](./elements-nodes/document-i/instance-methods.md#querySelectorAll) и в [Element](./elements-nodes/element-i/instance-methods.md#queryselectorall)
- поиск первого [querySelector в Element](./elements-nodes/document-i/instance-methods.md#querySelector)
