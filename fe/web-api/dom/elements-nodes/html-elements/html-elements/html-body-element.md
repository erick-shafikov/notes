# HTMLBodyElement

наследует [HTMLElement](../html-element-i.md)

# пользовательские dom свойства

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
