# element.style

Это объект, который соответствует тому, что написано в атрибуте style но не в CSS!!!
elem.style.width = "100px" работает так же как наличие в атрибуте style строки width: 100px
Для свойств из нескольких слов

background-color => elem.style.backgroundColor
z-index => elem.style.zIndex
border-left-with => elem.style.borderLeftWidth

```js
document.body.style.backgroundColor = prompt("background color?", "green");
```

стили с браузерным прификсом
-moz-border-radius => button.style.MozBorderRadius = "5px";
-webkit-border-radius => button.style.WebkitBorderRadius ="5px";

# сброс стилей

при необходимости добавить свойство стиля а позже его убрать, то можно присвоить свойству пустую строку

```js
document.body.style.display = "none"; //скрыть
setTimeout(() => (document.body.style.display = ""), 1000); //возврат к нормальному состоянию
```

div.style – это объект, доступный только для чтения, для задания нескольких стилей

# cssText

style.cssText - позволяет вставить css Вставка стилей как текстовый атрибут

```js
et top = /* сложные расчёты */;
let left = /* сложные расчёты */;

// полная перезапись стилей elem, используем =
elem.style.cssText = `
  top: ${top};
  left: ${left};
`;

// добавление новых стилей к существующим стилям elem, используем +=
elem.style.cssText += `
  top: ${top};
  left: ${left};
`;

```

```html
<div id="div">Button</div>

<script>
    //перезапись флаг important
    div.style.cssText = "
      color:red !important;
      background-color: yellow;
      width: 100px;
      text-align:center
    ;"

  //добавление
    div.style.cssText += "
      color:red !important;
      background-color: yellow;
      width: 100px;
      text-align:center
    ;"

    alert(div.style.cssText); //выведет весь стиль элемента
</script>
```

При отсутствии добавления единиц измерения присвоение игнорируется

```js
document.body.style = 20; //проигнорирует
```

# Вычисляемые стили: getComputedStyle

При необходимости узнать размер, отступы, цвет элемента из CSS, а не только из атрибута style. Свойство Style оперирует только значением атрибута style бtз учета CSS-каскада. Есть два типа значений стиля:

- Вычисленное (computed) - это вычисленные значения после применения всех CSS правил но в относительных единицах, если такие есть rem, em
- Окончательное (resolved) - это значения в пикселях

```html
<head>
  <!-- из-за того, что стиль описан в глобальном стиле мы не сможем прочитать значения -->
  <style>
    body {
      color: red;
      margin: 5px;
    }
  </style>
</head>
<body>
  красный текст
  <script>
    alert(document.body.style.color); //пусто
    alert(document.body.style.marginTop); //пусто
  </script>
</body>
```

синтаксис getComputedStyle(element, [pseudo]) - результат вызова – объект со стилями похожий на elem.style
element – элемент для которого нужно получить значение
pseudo – указывается, если нужен стиль псевдоэлемента

```html
<head>
  <style> body {color:red; margin: 5px} </style>
</head>
<body>

  <script>
    let computedStyle = getComputedStyle(document.body)
    alert( computedStyle.marginTop); //5px
    alert( computedStyle.color); //rgb(255, 0, 0)
  <script>
```

Вычисленное (computed) значение – это, то которое получено после применения всех CSS правил и CSS свойств наследования в относительных величинах
Окончательное(resolved) – непосредственно применяемое к элементу
getComputedStyle – возвращает окончательное значение стиля

getComputedStyle – требует полного наименования свойства

стили примененные к посещенным ссылкам - игнорируются
