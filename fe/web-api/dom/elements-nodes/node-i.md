# Node

наследует [EventTarget](../events/event-target-i.md)

# свойства экземпляра

- baseURI - абсолютный базовый url узла при обработке img src атрибут(readonly)
- childNodes ⇒ Node[] возвращает потомков (readonly)

# firstChild

⇒ Node возвращает первого потомка (readonly)

```js
// задача список потомков в дереве

let lis = document.getElementsByTagName("li");
//псевдо массив с li-элементами
for (let li of lis) {
  //для каждого элемента с псевдо массива
  let sum = li.getElementsByTagName("li").length;
  // присвоить сумме количество вложенных li
  if (!sum) continue;
  //если li равно нулю то перейти к следующей итерации
  li.firstChild.data += "[" + sum + "]";
  // если li не равно нулю то добавить
}
```

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

# nodeValue/data

вернет value если есть у узла. Свойство innerHTML есть только у узлов-элементов

```html
 
<body>
      Привет    
  <!-- Комментарий -->
     
  <script>
    let text = document.body.firstChild;
    alert(text.data); //Привет
    let comment = text.nextSibling;
    alert(comment.data); //Комментарий
  </script>
   
</body>
```

- ownerDocument - document
- parentElement ⇒ Node родительский
- parentNode ⇒ Node родительский
- previousSibling ⇒ Node соседа перед
- textContent ⇒ текстовое значение node

## textContent

предоставляет доступ к тексту за вычетом всех тегов

```html
<div id="news">
  <h1>Срочно в номер!</h1>
  <p>Марсиане атаковали человечество</p>
  <p></p>
  <div>
    <script>
      alert(news.textContent); //Срочно в номер! Марсиане атаковали человечество
    </script>
  </div>
</div>
```

- - разница с innerText:
- - - textContent получает содержимое всех элементов, включая script style, тогда как innerText этого не делает.
- - - innerText умеет считывать стили и не возвращает содержимое скрытых элементов, тогда как textContent этого не делает.
- - - Метод innerText позволяет получить CSS, а textContent — нет.

# методы экземпляра

## appendChild, insertBefore, replaceChild, removeChild

- parentElem.appendChild(node) – добавляет node в конце дочерних элементов parentElem
- parentElem.insertBefore(node, nextSibling) – вставляет node перед nextSibling в parentElement
- parentElem.replaceChild(node, oldChild) заменяет oldChild на node среди дочерних элементов parentElem
- parentElem.removeChild(node) - ⇒ Node, удаляет node из parentElem,

## cloneNode

клонирует узел
elem.cloneNode(true) создает глубокий клон элемента со всеми атрибутами и дочерними элементами
elem.cloneNode(false) создает клон без дочерних элементов

```html
<style>
.alert {}
</style>
<div class="alert" id="div"><strong>Всем привет!Вы прочитали важное сообщение</div>
<script>
  let div2=div.cloneNode(true);//скопировать элемент div в div2
  div2.querySelector("strong").innerHTML = "Всем пока!"; //изменить все что в strong
  div.after(div2); //поставить div2 после div1
  //Всем привет! Вы прочитали важное сообщение
  //Все пока! Вы прочитали важное сообщение

</script>

```

<!--  -->

- compareDocumentPosition(otherNode) ⇒ битовую маску расположения элемента
- contains(otherNode) ⇒ boolean является otherNode дочерним
- getRootNode() ⇒ HTMLDocument или iframe
- hasChildNodes() ⇒ boolean если дочерние узлы или нет
- isDefaultNamespace(namespaceURI) ⇒ boolean если namespaceURI является NS данного узла
- isEqualNode(otherNode) ⇒ boolean совпадает ли otherNode с node
- isSameNode(otherNode) ⇒ boolean совпадает ли ссылка otherNode с node
- lookupNamespaceURI() ⇒ пространство имен
- lookupPrefix() ⇒ префикс NS
- normalize() - нормализует узел (убирает пустые имена)

## события

- selectstart - сработает при выделении
