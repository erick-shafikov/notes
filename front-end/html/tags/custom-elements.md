<!-- slot ------------------------------------------------------------------------------------------------------------------->

# slot (str)

именованный слот

Атрибуты:

- name

```html
<template id="element-details-template">
  <style>
    details {
      font-family: "Open Sans Light", Helvetica, Arial, sans-serif;
    }
    .name {
      font-weight: bold;
      color: #217ac0;
      font-size: 120%;
    }
    h4 {
      margin: 10px 0 -8px 0;
      background: #217ac0;
      color: white;
      padding: 2px 6px;
      border: 1px solid #cee9f9;
      border-radius: 4px;
    }
    .attributes {
      margin-left: 22px;
      font-size: 90%;
    }
    .attributes p {
      margin-left: 16px;
      font-style: italic;
    }
  </style>
  <details>
    <summary>
      <code class="name"
        >&lt;<slot name="element-name">NEED NAME</slot>&gt;</code
      >
      <i class="desc"><slot name="description">NEED DESCRIPTION</slot></i>
    </summary>
    <div class="attributes">
      <h4>Attributes</h4>
      <slot name="attributes"><p>None</p></slot>
    </div>
  </details>
  <hr />
</template>
```

<!-- template ------------------------------------------------------------------------------------------------------------------>

# template (HTML5)

Инкапсулирует html элементы

```html
<table id="producttable">
  <thead>
    <tr>
      <td>UPC_Code</td>
      <td>Product_Name</td>
    </tr>
  </thead>
  <tbody>
    <!-- данные будут вставлены сюда -->
  </tbody>
</table>

<template id="productrow">
  <tr>
    <td class="record"></td>
    <td></td>
  </tr>
</template>
```

```js
// Убеждаемся, что браузер поддерживает тег <template>,
// проверив наличие атрибута content у элемента template.
if ("content" in document.createElement("template")) {
  // Находим элемент tbody таблицы
  // и шаблон строки
  var tbody = document.querySelector("tbody");
  var template = document.querySelector("#productrow");

  // Клонируем новую строку и вставляем её в таблицу
  var clone = template.content.cloneNode(true);
  var td = clone.querySelectorAll("td");
  td[0].textContent = "1235646565";
  td[1].textContent = "Stuff";

  tbody.appendChild(clone);

  // Клонируем новую строку ещё раз и вставляем её в таблицу
  var clone2 = template.content.cloneNode(true);
  td = clone2.querySelectorAll("td");
  td[0].textContent = "0384928528";
  td[1].textContent = "Acme Kidney Beans 2";

  tbody.appendChild(clone2);
} else {
  // Иной способ заполнить таблицу, потому что
  // HTML-элемент template не поддерживается.
}
```

```html
<div id="container"></div>

<template id="template">
  <div>Click me</div>
</template>
```

```js
const container = document.getElementById("container");
const template = document.getElementById("template");

function clickHandler(event) {
  event.target.append(" — Clicked this div");
}

const firstClone = template.content.cloneNode(true);
firstClone.addEventListener("click", clickHandler);
container.appendChild(firstClone);

const secondClone = template.content.firstElementChild.cloneNode(true);
secondClone.addEventListener("click", clickHandler);
container.appendChild(secondClone);
```
