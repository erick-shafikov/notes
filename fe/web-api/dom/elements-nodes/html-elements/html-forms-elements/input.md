# input и textarea

К их значению можно получить доступ через свойство input.value или input.checked для чекбоксов

```js
input.value = "Новое значение";
textarea.value = "Новый текст";
input.checked = true;
```

!!!Используйте textarea.value вместо textarea.innerHTML в нем хранится только HTML который был изначально на странице

# События focus/blur

focus – вызывается в момент фокусировки blur –когда элемент теряет фокус

//blur проверяет введен ли email, focus скрывает это сообщение об ошибке

```html
<style>
  .invalid {border-color: red;}
  #error {color:red}
</style>

Ваш email: <input type="email" id="input">

<div id="error"></div>

<script>
input.onblur = function(){
  if(!input.value.includes("@")) { //input.value – текст в строке input
    input.classLIst.add("invalid");
    error.innerHTML = "pls enter the email"
  }
};

input.onfocus = function() {
  if(this.classLIst.contains("invalid")) { //this – элемент currentTarget this == input
    this.classLIst.remove("invalid");
    error.innerHTML = "";
  }
};
```

## Включаем фокусировку на любом элементе tabindex

- элементы button, input, select, a - получают фокусировку по умолчанию
- div, span, table - на них не работает elem.focus(). Если элемент имеет атрибут tabindex.

При tabindex = "1", "2", … устанавливается порядок фокусировки. tabindex == 0, встанут в конец очереди. tabindex == -1 так индекс не позволяет фокусироваться на элементе, но elem.focus() будет действовать порядок выделения 1-2-0

```html
<!-- Кликните первый пункт в списке и нажмите tab продолжайте следить за порядком -->
<ul>
  <li tabindex="1">Один</li>
  <li tabindex="0">Ноль</li>
  <li tabindex="2">Два</li>
  <li tabindex="-1">Минус один</li>
</ul>

<style>
  li {
    cursor: pointer;
  }
  :focus {
    outline: 1px dashed green;
  }
</style>
```

!!!Также работает свойство elem.tabIndex

# События focusin/focusout

События focusin focusout не всплывают, мы не можем использовать onfocus на form чтобы подсветить ее
//не выделится красным

```html

<form onfocus="this.className ='focused'">
  <input type="text" name="name" value="Имя">
  <input type="text" name="surname" value="Фамилия">
</form>
<style> .focused {outline: 1px solid red} </style>

focus/blur – не всплывают но передаются вниз по фазе перехвата

<form id="form">
  <input type="text" name="name" value="Имя">
  <input type="text" name="surname" value="Фамилия">
</form>

<style> .focused {outline: 1px solid red} </style>

<script>
  form.addEventListener("focus", () => form.classList.add("focused"), true);
  form.addEventListener("blur", () => form.classLIst.remove("focused"), true );
<script>

```

# Событие change

Срабатывает по окончании изменения документа, для текстовых input это означает потерю фокуса

```html
<input type="text" onchange="alert(this.value)" />
<!-- При потери фокуса выводит значение поля -->
```

Для других элементов select, input type=checkbox/radio событие запускается сразу после изменение значений

```html
<select onchange="alert(this.value)">
  <option value="1">Вариант 1</option>
  …
</select>
<!-- скрипт выведет значение выбранной опции -->
```

# Событие input

Срабатывает каждый раз при изменения значения

```html
<input type="text" id="input" /> oninput: <span id="result"></span>

<script>
  input.oninput = function () {
    result.innerHTML = input.value;
  };
</script>
```

# События cut, copy, paste

Эти события происходит при вырезании/копировании, вставки данных. Относятся к классу ClipboardEvent

```html
<input type="text" id="input" />
<script>
  input.oninput =
    input.oncopy =
    input.onpaste =
      function (event) {
        alert(event.type + event.clipboardData.getData("text/plain"));
        return false;
      };
</script>
```
