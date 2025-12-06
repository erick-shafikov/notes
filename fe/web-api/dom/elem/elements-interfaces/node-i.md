# Node

наследует [EventTarget](./event-target-i.md)

# свойства экземпляра

- baseURI - абсолютный базовый url узла при обработке img src атрибут(readonly)
- childNodes ⇒ Node[] возвращает потомков (readonly)
- firstChild ⇒ Node возвращает первого потомка (readonly)
- isConnected ⇒ boolean прикреплен ли элемент к dom (read-only)
- lastChild ⇒ Node возвращает последнего потомка (readonly)
- nextSibling ⇒ Node соседа
- nodeName - название узла, определено для любых узлов Node
- nodeType - тип узла:
- - elem.nodeType == 1 для узлов – элементов
- - elem.nodeType == 3 для текстовых узлов
- - elem.nodeType == 9 Для объектов документа

  ```html
   
  <body>
    <!-- document - корневой элемент, element- узел dom -->
    <!-- комментарий-->
     
    <script>
      alert(document.body.firstChild.tagName); //undefined (не элемент)
      alert(document.body.firstChild.nodeName); // comment
      alert(document.tagName); //undefined (не элемент)
      alert(document.nodeName); // document
    </script>
     
  </body>
  ```

- nodeValue - вернет value если есть у узла
- ownerDocument - document
- parentElement ⇒ Node родительский
- parentNode ⇒ Node родительский
- previousSibling ⇒ Node соседа перед
- textContent ⇒ текстовое значение node
- - разница с innerText:
- - - textContent получает содержимое всех элементов, включая script style, тогда как innerText этого не делает.
- - - innerText умеет считывать стили и не возвращает содержимое скрытых элементов, тогда как textContent этого не делает.
- - - Метод innerText позволяет получить CSS, а textContent — нет.

# методы экземпляра

- appendChild(child) - вставит в конец списка дочерних
- cloneNode(node) - клонирует узел
- compareDocumentPosition(otherNode) ⇒ битовую маску расположения элемента
- contains(otherNode) ⇒ boolean является otherNode дочерним
- getRootNode() ⇒ HTMLDocument или iframe
- hasChildNodes() ⇒ boolean если дочерние узлы или нет
- insertBefore(newElement, referenceElement) - вставит newElement перед referenceElement
- isDefaultNamespace(namespaceURI) ⇒ boolean если namespaceURI является NS данного узла
- isEqualNode(otherNode) ⇒ boolean совпадает ли otherNode с node
- isSameNode(otherNode) ⇒ boolean совпадает ли ссылка otherNode с node
- lookupNamespaceURI() ⇒ пространство имен
- lookupPrefix() ⇒ префикс NS
- normalize() - нормализует узел (убирает пустые имена)
- removeChild(child) ⇒ Node удаляет и возвращает старую Node
- replaceChild(newChild, oldChild) ⇒ поменяет oldChild на newChild на узле

## события

- selectstart - сработает при выделении
