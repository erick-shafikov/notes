# Document Object Model

Объектная модель HTML документа, то как браузер представляет HTML страницу в виде объекта, в DOM дереве существует 12, такие как document, тэги, текстовые узлы, комментарии, атрибуты. Все что есть в HTML есть в DOM дереве.

Каждый узел дерева – объекты. Теги являются узлами – элементами head и body его дочерние элементы
- !!!Пробел и перевод строки перед тегом head игнорируются
- !!!Если записываем что-то после тега body браузер автоматически перемещает эту запись в конец body, то есть после </body> не может быть никаких пробелов

Каждый DOM-элемент соответствует встроенному классу

- Event – корневой абстрактный класс
- Node – основной для DOM узлов, обеспечивает функциональность для parentNode, nextSibling, childNOdes (это все геттеры)
- Element – базовый класс для DOM обеспечивает навигацию

# nodeType

elem.nodeType == 1 для узлов – элементов
elem.nodeType == 3 для текстовых узлов
elem.nodeType == 9 Для объектов документа

nodeName – определено для любых узлов Node
tagName – есть только у элементов Element

```html
 
<body>
  <!-- комментарий-->
   
  <script>
    alert(document.body.firstChild.tagName); //undefined (не элемент)
    alert(document.body.firstChild.nodeName); // comment
    alert(document.tagName); //undefined (не элемент)
    alert(document.nodeName); // document
  </script>
   
</body>
```

document - корневой элемент
element- узел dom

# DOM-свойства

```html
<!-- Можем создать свойство для document.body -->
<body id="page">
  <!-- body.id = "page" в DOM -->

  <script>
    document.body.myData = {
      name: "Cesar",
      title: "Imperator",
    };

    alert(document.body.myData.title); //Imperator

    document.body.syaTagName = function () {
      alert(this.tagName);
    };

    document.body.sayTagName(); //BODY (значение this в это методе будет document.body)

    Element.prototype.sayHI = function () {
      alert(`Hello, i"m ${this.tagName}`);
    };

    document.documentElement.sayHI(); // Hello, I"m HTML
    document.body.sayHi(); //Hello, I"m BODY
  </script>
</body>
```

# оптимизация работы с DOM:

- В скриптах минимизируйте любую работу с DOM. Кэшируйте всё: свойства, объекты, если подразумевается повторное их использование. - При сложных манипуляциях разумно работать с «offline» элементом (т.е. который находится не в DOM, а в памяти), с последующим помещением его в DOM.
- Упрощать css – селекторы, чем меньше вложенность тем лучше
- Минимизировать любую работу DOM, изменять однократно
- Для изменений стилей лучше оперировать атрибутом class
- Анимировать лучше элементы позиционировать абсолютно или фиксировано, не изменять геометрию узлов
- откладывать невидимые изменения
- использовать web workers для больших и сложных задач
- использование requestAnimationFrame
