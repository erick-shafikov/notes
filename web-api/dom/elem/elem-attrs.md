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

# HTML атрибуты

Когда у элемента есть id или другой стандартный атрибут создается соответствующее свойство, но если атрибут нестандартный, то этого не происходит. HTML атрибуты регистронезависимые значения – строки

```html
<body id="test" something="non-standard">
  <script>
    alert(document.body.id); //test
    alert(document.body.something); //undefined
    // методы работы с атрибутами
    elem.hasAttribute(name); //проверяет наличие атрибута
    elem.getAttribute(name); //получает значение атрибута
    elem.setAttribute(name, value); //устанавливает значение атрибута
    elem.removeAttribute(name); //удаляет атрибут
    elem.attributes; //коллекция объектов
  </script>
</body>
```

Синхронизация между атрибутами и свойствами

```html
<input />

<script>
  let input = document.querySelector("input");
  input.setAttribute("id", "id");
  alert(input.id); //id
  input.id = "newId";
  alert(input.getAttribute("id")); //newId
</script>

<!-- Исключение для input.value -->

<input />

<script>
  let input = document.querySelector("input");
  input.setAttribute("value", "text");
  alert(input.value); //text

  input.value = "newValue";
  alert(input.getAttribute("Value")); //text не обновилось
</script>
```

# DOM свойства типизированы

DOM свойства не всегда являются строками

```html
<input id="input" type="checkbox" checked />
<!--checkbox-->

<script>
  alert(input.getAttribute("checked")); //значение пустая строка
  alert(input.checked); //значение свойства true
</script>

<!-- свойство style является объектом -->

<div id="div" style="color:red; font-size:120">Hello</div>
<script>
  alert(div.getAttribute("style")); //{color: red; font-size: 120%}

  alert(div.style); //[object CSSStyleDeclaration]
  alert(div.style.color); //red
</script>

<!-- Свойство href всегда содержит полный URL, даже если содержит относительный путь
или # -->

<a id="a" href="#hello">link</a>

<script>
  alert(a.getAttribute("href")); //#hello
  alert(a.href); //полный URL
</script>
```

# Нестандартные атрибуты

```html
<!-- Заполнить html соответствующими элементами помечать div чтобы показать что здесь поле name -->
<div show-info="name"></div>
<!-- здесь age -->
<div show-info="age"></div>

<script>
  let user = {
    name: "Pete",
    age: 25
  };

  for(let div of document.querySelectorAll["show-info"]) {
   let field = div.getAttribute("show-info");
   div.innerHTML = user[field];
  }
<script>

```

```html
<style>
  .order[order-state="new"] {
    color: green;
  }

  .order[order-state="pending"] {
    color: blue;
  }

  .order[order-state="canceled"] {
    color: red;
  }
</style>

<div class="order" order-state="new">A new order</div>
<div class="order" order-state="pending">A pending-order</div>
<div class="order" order-state="canceled">A canceled order</div>
```

все атрибуты data- зарезервированы для использования программистами

если у elem есть атрибут data-about то обратиться к нему можно как elem.dataset.about

```html
<body data-about="Elephants">
<script>
  alert(document.body.dataset.about); //Elephants
<script>

```

атрибуты состоящие из нескольких слов data-order-state становится свойствами dataset.orderState

```html
<style>
.order[order-state="new"] {color: green;}
.order[order-state="pending"] {color: blue;}
.order[order-state="canceled"] {color: red;}
</style>

<body>
  <div class="order" data-order-state="new">A new order</div>
  <div class="order" data-order-state="pending">A pending-order</div>
  <div class="order" data-order-state="canceled">A canceled order </div>
</body>

<script>
  alert(order.dataset.orderState); //new
<script>

```
