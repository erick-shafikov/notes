# Свойства и методы формы

## Навигация

Формы в документе входят в специальную коллекцию document.forms
document.forms.my – форма с именем my, name === my
document.forms[0] – первая форма в документе
Любой элемент доступен в именованной коллекции form.elements (т.е либо document.forms.[name] или form.elements.[name])

```html
<form name="my">
  <input name="one" value="1" />
  <input name="two" value="2" />
</form>

<script>
  let form = document.forms.my;
  let elem = form.elements.one;
  alert(elem.value); //1
</script>
```

Могут быть несколько элементов с одним и тем же именем , тогда form.elements[name] является коллекцией

```html
<form>
  <input type="radio" name="age" value="10" />
  <input type="radio" name="age" value="20" />
  <script>
    let forms = document.forms[0];
    let ageElem = form.elements.age;

    alert(ageElem[0]); //[object HTMLInputElement]
  </script>
</form>
```

Все элементы управления формы доступны в коллекции form.elements

# fieldset как под-форма

Форма может содержать один или несколько элементов fieldset

```html
<body>
  <form id="form">
    <!-- открыть блок -->
   <fieldset name="userFields">
     <!-- подпись блока -->
      <legend>info</legend>
      <input name="login" type="text">
   </fieldset>
   </form>
  </body>

<script>
  alert(form.elements.login); //<input name="login">
  let fieldset = form.elements.userFields; //fieldset"ы доступны так же как элемент в elements
  alert(fieldset); //HTMLFieldsetElement
  alert(fieldset.elements.login == form.elements.login); //true
<script>

```

### Сокращенная форма записи form.name

мы можем получить доступ к элементу через form[index/name], то есть вместо form.elemnts.login == form.logins, минус в том, что если мы получаем элемент, а затем меняем его свойство name, то он все еще будет доступен под старым мененм (так же как и под новым)

```html
<form id="form">
  <input name="login" />
</form>

<script>
  alert(form.elements.login == form.login); //true. т к одинаковые <input>
  form.login.name = "username"; //меняем свойство name, теперь <input name="username">
  alert(form.elements.login); //undefined, так как его переименовали
  alert(form.elements.username); //input новый переименованный

  alert(form.username == form.login); //true а в form можем использовать и новое с старое имя
</script>
```

### Форма ссылается на элементы, а элементы ссылаются на форму

<form id="form">
  <input type="text" name="login">
</form>

<script>
  let login = form.login;
  alert(login.form); //HTMLFormElement
</script>

## input и textarea

К их значению можно получить доступ через свойство input.value или input.checkbox для чекбоксов

```js
input.value = "Новое значение";
textarea.value = "Новый текст";
input.checked = true;
```

!!!Используйте textarea.value вместо textarea.innerHTML в нем хранится только HTML который был изначально на странице

## Элементы формы

select имеет 3 свойства
select.options – коллекция из под-элементов <option>
select.value – значение выбранного в данный момент <option>
select.selectedIndex – номер выбранного <option>

они дают три способа установить значение в <select>
Найти соответствующий элмент <option> и установить в option.selected Значение true
установить в select.value – значение выбранного в данный момент <option>
установить в select.selectedIndex номер нужного <option>
//три варианта выбрать банан, как стартовый пункт в всплывающем меню

```html
<select id="select">
  <option value="apple">Яблоко</option>

  <option value="pear">Груша</option>
  <option value="banana">Банан</option>
</select>

<script>
  select.options[2].selected = true; //как правило такой вариант не выбирают
  select.selectedIndex = 2;
  select.value = "banana";
</script>
```

select позволяет нам выбрать несколько вариантов одновременно, если у него стоит атрибут multiple

```html
<select id="" select="" multiple>
  <option value="" blues="" selected>Блюз</option>
  <option value="" rock="" selected>Рок</option>
  <option value="" classic="">Классика</option>
</select>

<script>
  let selected = Array.from(select.option)
    .filter((option) => option.selected) //добавляет если true
    .map((option) => option.value); //меняет элемент массивов на value

  alert(selected);
</script>
```

### new Option

Синтаксис создания элемента option

```js
option = new Option(text.value, defaultSelected, selected);
```

text – текст внутри option
value – значение
defaultSelected – если true то ставится HTML атрибут selected
selected – если true то элемент option будет выбранным

```js
let option = new Option("Текст", "value"); //создаст <option value="value">Текст</option>
let option = new Option("Текст", "value", true, true); //такой же элемент только выбранный в списке
```

# Фокусировка

элемент получает фокус, год пользователь кликает по элементу или нажимает tab

## События focus/blur

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

элементы <button>, <input>, <select>, <a> - получают фокусировку по умолчанию

<div>, <span>, <table> - на них не работает elem.focus(). Если элемент имеет атрибут tabindex.
При tabindex = "1", "2", … устанавливается порядок фокусировки
tabindex = "0", встанут в конец очереди. tabindex = "-1" так индекс не позволяет фокусироваться на элементе, но elem.focus() будет действовать
//подрядок выделения 1-2-0
Кликните первый пункт в списке и нажмите tab продолжайте следить за порядком

```html
<ul>
  <li tabindex=""1"">Один</li>
  <li tabindex=""0"">Ноль</li>
  <li tabindex=""2"">Два</li>
  <li tabindex=""-1"">Минус один</li>
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

## События focusin/focusout

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

# События change, input, cut, copy, paste

## Событие change

Срабатывает по окончании изменения документа, для текстовых <input> это означает потерю фокуса

```html
<input type="text" onchange="alert(this.value)" />
<!-- При потери фокуса выводит -->
значение поля
```

Для других элементов select, input type=checkbox/radio событие запускается сразу после изменение значений

```html
<select onchange="alert(this.value)">
  <option value="1">Вариант 1</option>
  …
</select>
<!-- скрипт выведет значение выбранной опции -->
```

## Событие input

Срабатывает каждый раз при изменения значения

```html
<input type="text" id="input" /> oninput: <span id="result"></span>

<script>
  input.oninput = function () {
    result.innerHTML = input.value;
  };
</script>
```

## События cut, copy, paste

Эти события происходит при вырезании/копировании, вставки данных. Относятся к классу ClipboardEvent. event.preventDefault()

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

# Отправка формы: событие и метод submit

## Событие submit

Два способа отправить форму:
Нажать на кнопку <input type=“submit”> или <input type=“image”>
Нажать на Enter, находясь на каком-нибудь поле

Оба генерируют событие submit, можно вызывать event.preventDefault(), для предотвращения отправки формы

```html
<form onsubmit="alert('submit'); return false">
  <input type="text" value="Текс" />
  <input type="submit" value="Отправить" />
</form>
<!-- в данном примере сработает alert, но форма не будет отправлена -->
```

При отправке формы по нажатию enter в текстовом поле генерируется событие click на кнопке <input type=“submit”> //Нажатие на enter генерирует событие click

```html
<form onsubmit="alert('submit'); return false">
  <input type="submit" value="отправить" onclick="alert('клик')" />
</form>
<!-- сработает два alert и на событие submit и на событие onclick -->
```

### Метод submit

Что бы отправить форму вручную, нужно вызвать метод submit. form.submit()
!!!При этом событие submit Не генерируется

```js
let form = document.createElement("form");
form.action = "https://google.com/search";
form.method = "GET";
form.innerHTML = `<input name="q" value="test" />`;
document.body.append(form);
form.submit(); //откроется страница с поиском в гугле с запросом test
```

## BP. Модальное диалоговое окно с формой

```js
function showCover() { //функция для создания полупрозрачного div
  //показать полупрозрачный div чтобы затемнить страницу, форма располагается рядом, а не внутри него
  let coverDiv = document.createElement("div");
  cover.id = "cover-div";
  document.body.style.overflowY = "hidden"; //убираем возможность прокрутки страницы
  document.body.append("coverDiv")
}

function hideCover() { //функция для того чтобы убрать
  document.getElementById("cover-div").remove;
  document.body.style.overflowY = "";
}
  //основная функция с двумя параметрами в виде текста для модального окна(text) и фнкцией-calback
function showPrompt("text", callback) {

  showCover(); //добавляем полупрозрачную обертку
  let form = document.getElementById("prompt-form"); //находим форму внутри
  let container = document.getElementById("prompt-form-container"); //контейнер внутри которого находится форма
  document.getElementById("prompt-message").innerHTML = "text"; //вставляем текст, где должен быть текст
  form.text.value = "";

  function complete(value) { //функция вызываемая по завершению
   hideCover(); //убираем обертку
   container.style.display = "none"; //убираем контейнер с формой
   document.onkeydown = "null"; //убрать все события после завершения связанные с нажатием клавиш
   callback(value); //вызываем коллбек
  }}

  form.onsubmit = function(){ //фикция по отправке формы
    let value = form.text.value; //текст формы
    if(value == "") return false; //если его нет, то не делать ничего
    complete(value) //завершение функции
    return false //не отправлять ничего
  };

  form.cancel.onclick = function(){
   complete(null); //при отмене вызывать функцию с коллбеком null
  };

  document.onkeydown = function(e){
  if(e.key == "Escape") {
    complete(null); //при esc
  }
};
//для полей формы, если нажаты нужные клавиши
let lastElement = form.elements[form.elements.length-1];
let firstElement = form.elements[0];
lastElem.onkeydown = function(e) {
   if(e.key == "Tab" && e.shiftKey){
     firstElem.focus();
     return false;
}

firstElem.onkeydown = function(e) {
  if(e.key == "Tab" && e.shftKey) {
   lastElem.focus()
   return false;
}};


container.style.display="block";
form.elements.text.focus();
}

document.getElementById("show-button").onclick =
function() {
  showPrompt("введите-что-нибудь") ? function(value) {
   alert(«вы ввели»);}
};

```
