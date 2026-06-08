# поиск элементов в документе

# getElementsByClassName

возвращает все элементы, которые имеют данный CSS-класс

# querySelectorAll

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

# querySelector

```js
elem.querySelector("css-rule"); //возвращает первый элемент соответствующий CSS-селектору
elem.querySelectorAll("css-rule")[0] == elem.querySelector("css-rule");
```

# живые коллекции

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

# matches

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

# closest

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

# contains

elemA.contains(elemB) вернет true если elemB находится внутри elemA
