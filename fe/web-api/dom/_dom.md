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
