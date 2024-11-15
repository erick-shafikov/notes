# className и classList

класс как свойство className: elem.className соответствует атрибуту класс

```html
<body class="main page">
   
  <script>
    alert(document.body.className); //main page
  </script>
</body>
```

!!!При присваивании заменяет полностью строку класса

classList - основной метод работы с классами

```js
elem.classList.add("class"); //– добавить класс
elem.classList.remove("class"); //- удаляет класс
elem.classLIst.toggle("class"); //– добавить класс, если его нет иначе удалить
elem.classList.contains("class"); //- проверка на наличие
```

```html
<body class="main page">
   
  <script>
    document.body.classList.add("article");
    alert(document.body.className); //main page article

    for (let name of document.body.classList) {
      alert(name); //name, page, article
    }
  </script>
</body>
```
