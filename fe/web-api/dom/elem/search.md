# поиск элементов в документе

<!-- getElementById ------------------------------------------------------------------------------------------------------------------------>

# getElementById

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

<!-- getElementsByTagName ------------------------------------------------------------------------------------------------------------------>

# getElementsByTagName

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

<!-- getElementsByClassName ---------------------------------------------------------------------------------------------------------------->

# getElementsByClassName

возвращает все элементы, которые имеют данный CSS-класс

<!-- querySelectorAll ---------------------------------------------------------------------------------------------------------------------->

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

<!-- querySelector ------------------------------------------------------------------------------------------------------------------------->

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

<!-- matches ------------------------------------------------------------------------------------------------------------------------------->

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

<!-- closest ------------------------------------------------------------------------------------------------------------------------------->

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

<!-- contains ------------------------------------------------------------------------------------------------------------------------------>

# contains

elemA.contains(elemB) вернет true если elemB находится внутри elemA
