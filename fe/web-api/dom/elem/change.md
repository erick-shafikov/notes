<!-- innerHtml ------------------------------------------------------------------------------------------------------------------------------->

# innerHtml

позволяет получить HTML – содержимое элемента в виде строки, так же выступает сеттером

```html
 
<body>
     
  <p>Параграф</p>
     
  <div>DIV</div>
   
  <script>
    alert(document.body.innerHTML); //читаем текущее содержимое
    document.body.innerHTML = "Новый BODY!"; //меняем содержимое
  </script>
</body>
<!-- При вставке некорректного HTML браузер исправит ошибки -->
<body>
   
  <script>
    document.body.innerHTML = "<b>тест"; //забыли закрыть тег
    alert(document.body.innerHTML); //<b>тест</b> исправлено
  </script>
</body>
```

!!!но если вставить тег script - он становится частью HTML но не запускается
!!!innerHTML += осуществляет перезапись, старое содержимое удаляется на его место встает новая запись

<!-- outerHTML ------------------------------------------------------------------------------------------------------------------------------->

# outerHTML

свойство outerHTML содержит HTML элемента целиком. Это как innerHTML плюс сам элемент

```html
<div id="elem">Привет<b>Мир</b></div>

<script>
  alert(elem.outerHTML); //<div id="elem">Привет<b>Мир</b></div>
</script>
<!-- в отличие от innerHTML запись в outerHTML не изменяет элемент, вместо этого
элемент заменяется во внешнем контексте -->
<div>Привет, мир!</div>

<script>
  let div = document.querySelector("div");
  div.outerHTML = "<p>Новый элемент</p>";
  // 1. div был удален из документа
  //2. Вместо него вставлен другой HTML
  //3. в div осталось старое значение
  alert(div.outerHTML); //<div>Привет, мир!</div>
</script>
```

<!-- textContent ----------------------------------------------------------------------------------------------------------------------------->

# textContent

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

<!-- nodeValue/data -------------------------------------------------------------------------------------------------------------------------->

# nodeValue/data

Свойство innerHTML есть только у узлов-элементов

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

# Изменение документа

```js
document.createElement(tag); //- создает элемент с заданным тэгом
let div = document.createElement("div");

document.createTextNode(text); //– создает текстовый узел с заданным текстом
let textNode = document.createTextNode("А вот и я");

// Создать элемент с текстом
let div = document.createElement("div");
div.className = "alert";
div.innerHTML = "<strong>Всем привет</strong> Вы прочитали важное сообщение"; //создали элемент, но пока он не является частью документа
```

<!-- Методы вставки -------------------------------------------------------------------------------------------------------------------------->

# Методы вставки

```html
<style>
  .alert {
  }
</style>

<script>
  //создаст div элемент со стилем div и текстом
  let div = document.createElement("div");
  div.className = "alert";
  div.innerHTML = "<strong>….";

  document.body.append(div);
</script>
```

- node.append(content) – добавляет узлы или строки в конец node
- node.prepend(content) – вставляет узлы или строки в начало node
- node.before(content) – вставляет узлы или строки до node
- node.after(content) – вставляет узлы или строки после node
- node.replaceWith(content) – заменяет node заданными узлами или строками

в качестве контента могут быть строки или элементы

```html
<ol id="ol">
  <li>0</li>
  <li>1</li>
  <li>2</li>
</ol>

<script>
ol.before("before"); //вставить строку before перед ol
ol.after("after"); //вставить строку after после ol

let liFirst = document.createElement("li"); //создать элемент li присвоить переменной liFirst
liFirst.innerHTML = "prepend"; //содержимое liFirst строке prepend
ol.prepend(liFirst); //подставить перед ol

let liLast = document.createElement("li");
liLast.innerHTML = "append";
ol.append(liLast);
</script>

<!-- Методы могут вставлять несколько узлов -->
<div id="div"></div>
<script>
  div.before("<p>Привет<p>", document.createElement("hr")) //строчка привет с подчеркиванием;
<script>

```

<!-- Методы вставки insertAdjacentHTML/text/Element ------------------------------------------------------------------------------------------>

# Методы вставки insertAdjacentHTML/text/Element

Чтобы вставить HTML как HTML

elem.insertAdjacentHTML(where, html)

where принимает значения:

- beforebegin – вставить html перед elem
- afterbegin – вставить html в начало
- beforend – вставить html в конец elem
- afterend – вставить html после elem

```html
<div id="div"></div>
<script>
  div.insertAdjacentHTML("beforebegin", "<p>Привет</p>");
  div.insertAdjacentHTML("afterbegin", "<p>Пока<p>");
<script>
```

elem.insertAdjacentText(where, text) – вставляет текст вместо html
elem.insertAdjacentElement(where, element) – вставляет element

```html
<style>
  .alert {
  }
</style>
<script>
  document.insertAdjacentHTML(
    "afterbegin",
    '<div class="alert"><strong>Всем</strong>Вы прочитали важное сообщение</div>'
  );
</script>
```

<!-- remove ---------------------------------------------------------------------------------------------------------------------------------->

# Удаление узлов. elem.remove()

```html
<style>
.alert {}
</style>
<script>
  let div = document.createElement("div");
  div.className = "alert";
  div.innerHTML = "Строка"

  document.body.append("div");
  setTimeout(()=> div.remove(), 1000);
</script>

<!-- Все методы вставки удаляют узлы со старых мест, если нужно переместить, то не нужно его удалять -->
<div id="first">Первый</div>
<div id="second">Второй</div>
<script>
  second.after(first); // after, before, prepend, append
<script>
```

<!-- Клонирование узлов clone ---------------------------------------------------------------------------------------------------------------->

# Клонирование узлов

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

<!-- DocumentFragment ------------------------------------------------------------------------------------------------------------------------>

# DocumentFragment

является оберткой для передачи списков узлов, мы можем добавить к нему другие узлы

getListContent генерирует фрагмент с элементами <li>, которые позже вставляются в <ul>

```html
<ul id="ul"></ul>

<script>
  function getListContent() {
    let fragment = new DocumentFragment(); //создаем вставляемый фрагмент

    for (let i = 1; i <= 3; i++) {
      let li = document.createElement("li"); //создаем li элемент
      li.append(i); //создать номера в списке
      fragment.append(li); //вставляем во фрагмент элемент li c номером
    }

    return fragment; //возвращаем результат функции
  }

  ul.append(getListContent()); //он исчезнет (не добавится) вместо него вставится список с тремя li
</script>
```

# Устаревшее методы вставки и удаления

- parentElem.appendChild(node) – добавляет node в конце дочерних элементов parentElem
- parentElem.insertBefore(node, nextSibling) – вставляет node перед nextSibling в parentElement
- parentElem.replaceChild(node, oldChild) заменяет oldChild на node среди дочерних элементов parentElem
- parentElem.removeChild(node) удаляет node из parentElem
- document.write(html) – записывает html на страницу может быть сгенерирован по ходу

<!-- Задачи -------------------------------------------------------------------------------------------------------------------------------->

# Задачи

## задача очистить элемент

```js
function clear(elem) {
  //не будет работать потому что каждый вызов remove() сдвинет коллекцию
  for (let i = 0; i < elem.childNodes.length; i++) {
    elem.childNodes[i].remove();
  }
}

function clear(elem) {
  //будет работать
  while (elem.firstChild) {
    elem.firstChild.remove();
  }
}

function clear(elem) {
  //также будет работать
  elem.innerHTML = "";
}
```

## задача дерево из объекта

Задача превратить объект в документ Вариант 1 с помощью DOM

```html
<body>
    
  <div id="container"></div>
      
</body>
<script>
  "use strict";
  let data = {
    Рыбы: {
      форель: {},
      лосось: {},
    },
    Деревья: {
      Огромные: {
        секвойя: {},
        дуб: {},
      },
      Цветковые: {
        яблоня: {},
        магнолия: {},
      },
    },
  };
  function createTree(container, obj) {
    container.append(createTreeDom(obj));
  }

  function createTreeDom(obj) {
    if (!Object.keys(obj).length) return;
    //возвращать undefined при пустом объекте
    let ul = document.createElement("ul");
    //создаем Ul-элемент
    for (let key in obj) {
      let li = document.createElement("li"); //создаем li
      li.innerHTML = key;
      //внутрь li упаковываем key из цикла
      let childrenUL = createTreeDom(obj[key]); //рекурсивный вызов функции, если есть вложенные объекты
      if (childrenUL) {
        //при последнем шаге
        li.append(childrenUL); //добавить
      }
      ul.append(li);
    }
    return ul;
  }
  let container = document.getElementById("container");
  createTree(container, data);
</script>
```

Вариант 2 с помощью строк

```js
function createTree(container, obj) {
  container.innerHTML = createTreeText(obj);
  //упаковать в контейнер результат функции createTree
}
function createTreeText(obj) {
  let li = ""; //пустая строку в переменной li
  let ul; //объявим перченную ul
  for (let key in obj) {
    //для каждого свойства объекта
    li += "<li>" + key + createTreeText(obj[key]) + "</li>";
    // присвоить li значение ключа и рекурсивно вызвать функцию, li ожидает до последнего шага вложенности и не присваивается
  }
  if (li) {
    ul = "<ul>" + li + "</ul>"; //дошли до последнего шага вложенности, произошло присвоение переменной li, и все вложенные li оборачиваются в ul
  }
  return ul || ""; //функция возвращает ul или пустую строку на итерациях с пустым объектом
}
createTree(container, data);
```

## задача список потомков в дереве

```html
<script>
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
</script>
```

## Сортировка таблицы

```js
let sorted = Array.from(table.rows) //ряды таблицы в массив
  .slice(1) //первая строка не нужна, начать сортировку со второй
  .sort((rowA, rowB) =>
    rowA.cells[0].innerHTML > rowB.cells[0].innerHTML ? 1 : -1
  ); //сортируем таблицу , где cells внутри tr
table.tBodies[0].append(...sorted);
```
