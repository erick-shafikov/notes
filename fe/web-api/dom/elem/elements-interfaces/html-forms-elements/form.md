# Свойства и методы формы

# Навигация

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

# Сокращенная форма записи form.name

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

# Форма ссылается на элементы, а элементы ссылаются на форму

<form id="form">
  <input type="text" name="login">
</form>

<script>
  let login = form.login;
  alert(login.form); //HTMLFormElement
</script>

# Фокусировка

элемент получает фокус, год пользователь кликает по элементу или нажимает tab

# Метод submit

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

# Событие submit

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

# bps

# bps. Модальное диалоговое окно с формой

```js
function showCover() {
  //функция для создания полупрозрачного div
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
  if(e.key == "Tab" && e.shiftKey) {
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
