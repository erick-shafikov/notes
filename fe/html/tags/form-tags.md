<!-- button --------------------------------------------------------------------------------------------------------------------->

# button

Допустимое содержимое - текстовый контент

Атрибуты:

- autofocus - будет ли кнопка автоматически сфокусирована после загрузки страницы
- disabled
- form - содержит id формы в которую может быть даже и не вложен, если нет, то связан с родительским
- formaction - ссылка на обработчик формы, то есть action
- formenctype:
- - application/x-www-form-urlencoded - по умолчанию
- - multipart/form-data - если type file у input
- - text/plain
- formmethod: get, post
- formnovalidate - без валидации
- formtarget: \_self, \_blank, \_parent, \_top - где отображать ответ формы, если type === submit
- name - имя которое отправится с данными формы
- type:
- - submit - значение по умолчанию
- - reset - удалит данные с формы
- - button - нет поведения по умолчанию
- - menu - открывает меню с помощью [menu](#menu) элемента
- - value - начально значение кнопки

 <!-- datalist ----------------------------------------------------------------------------------------------------------------->

# datalist (block, HTML5)

отображает всплывающий tooltip с опциями, непосредственно связан с input по id

```html
<label for="ice-cream-choice">Choose a flavor:</label>
<!-- связь с id list="ice-cream-flavors" -->
<input list="ice-cream-flavors" id="ice-cream-choice" name="ice-cream-choice" />
<!-- связь с list id="ice-cream-flavors" -->
<datalist id="ice-cream-flavors">
  <option value="Chocolate"></option>
  <option value="Coconut"></option>
  <option value="Mint"></option>
  <option value="Strawberry"></option>
  <option value="Vanilla"></option>
</datalist>
```

<!-- fieldset legend----------------------------------------------------------------------------------------------------------->

# fieldset, legend (block)

fieldset применяется для создания заголовка группы элементов формы. Группа элементов обозначается в браузере с помощью рамки, а текст, который располагается внутри контейнера legend, встраивается в эту рамку.

Атрибуты fieldset:

- disabled - для всех полей формы
- form - id формы
- name - группы

Атрибуты legend

- accesskey - Переход к группе элементов формы с помощью комбинации клавиш
- align - Определяет выравнивание текста.
- title - Добавляет всплывающую подсказку к тексту заголовка.

```html
<!DOCTYPE html>

<html>
  <head>
    <meta charset="utf-8" />
    <title>Тег LEGEND</title>
  </head>

  <body>
    <fieldset>
      <legend>Работа со временем</legend>

      <p>
        <input type="checkbox" />
        создание пунктуальности (никогда не будете никуда опаздывать);<br />
        <input type="checkbox" />
        излечение от пунктуальности (никогда никуда не будете торопиться);<br />
        <input type="checkbox" /> изменение восприятия времени и часов.
      </p>
    </fieldset>
  </body>
</html>
```

<!-- form ---------------------------------------------------------------------------------------------------------------------->

# form (block)

Атрибуты:

- accept-charset - кодировки на сервер всегда "UTF-8"
- action - урл куда отправлять данные формы. В качестве обработчика может выступать CGI-программа или HTML-документ, который включает в себя серверные сценарии (например, Parser). После выполнения обработчиком действий по работе с данными формы он возвращает новый HTML-документ. Может быть переопределено в button
  Если атрибут action отсутствует, текущая страница перезагружается, возвращая все элементы формы к их значениям по умолчанию.
- autocomplete: off, on
- enctype: application/x-www-form-urlencoded, multipart/form-data, text/plain (HTML5)
- method:
- - post - посылает на сервер данные в запросе браузера. Это позволяет отправлять большее количество данных, чем доступно методу get, поскольку у него установлено ограничение в 4 Кб. Большие объемы данных используются в форумах, почтовых службах, заполнении базы данных, при пересылке файлов и др., значения добавляются в тело
- - get - Пары «имя=значение» присоединяются в этом случае к адресу после вопросительного знака и разделяются между собой амперсандом (символ &). Удобство использования метода get заключается в том, что адрес со всеми параметрами можно использовать неоднократно, сохранив его, например, в закладки браузера, а также менять значения параметров прямо в адресной строке.
- - dialog - если внутри dialog
- name - уникальное значение
- novalidate
- target - где отобразить ответ (\_self, \_blank, \_parent, \_top)

## action="mailto:"

Форма для отправки письма. В качестве обработчика можно указать адрес электронной почты, начиная его с ключевого слова mailto:.

```html
<form action="mailto:vlad@htmlbook.ru" enctype="text/plain">
  <p><input type="submit" value="Написать письмо" /></p>
</form>

<form action="mailto:shafikov_erick@mail.ru" method="post" enctype="text/plain">
  <!-- при нажатии на submit отправляет на форму написания письма   -->
  <label>Your Name:</label>  
  <input type="text" name="yourName" value="" /><br />
    <label>Your Email</label>  
  <input type="email" name="yourEmail" value="" /><br />
    <label>Your message</label><br />
    <textarea name="yourMessage" rows="10" cols="30"></textarea><br />//  
  <input type="submit" name="" />
</form>
```

<!-- input ---------------------------------------------------------------------------------------------------------------------->

# input

Атрибуты:

- глобальные
- accept - какие файлы может принимать input
- - "audio/", "video/", "image/"", "image/png", ".png"
- accesskey - управление фокусом
- mozactionhint - определяет кнопку на моб телефонах go, done, next, search, и send
- autocomplete: "off", "on" или какая-либо строка через пробел
- autofocus
- autosave - оставлять значение в строке, если type === search
- checked - для radio и checkbox
- disabled
- form - значение id формы, с которой форма связана
- formaction - uri
- formenctype: application/x-www-form-urlencoded, multipart/form-data, text/plain
- formmethod: get, post
- formnovalidate
- formtarget: \_self, \_blank, \_parent, \_top
- height, width - при type === image будет отображаться картинка-placeholder
- inputmode: verbatim, latin, latin-name, latin-prose, full-width-latin, kana, katakana, numeric, tel, email, url
- list - id элемента datalist
- max
- maxlength
- min
- minlength
- multiple - для email, file количество допустимых значений
- name - имя, которое идет в паре со значением
- pattern - regex, для паттерна
- placeholder
- readonly
- required - в css (:optional, :required)
- selectionDirection
- size
- spellcheck
- src - если type === img, То покажет placeholder
- step - если type == numeric, datetime
- tabindex
- type:
- - button - простая кнопка без определенного поведения (предпочтительней просто button)

```html
<!-- без value - пустая кнопка -->
<input class="styled" type="button" value="Add to favorites" />
```

- - checkbox - состояние может быть indeterminate - горизонтальная линия на чекбоксе
- - - Атрибуты:
- - - - checked - значение по умолчание, не отвечает за текущее состояние

```html
<!-- если чб не нажат, то в форме значение checkboxName === on, если нажат то checkboxName === checkboxValue -->
<input
  type="checkbox"
  id="checkboxName"
  name="checkboxName"
  value="checkboxValue"
/>

<!-- если в форме 2 чб с одинаковым именем, то на сервер отправится значение checkbox=value1&interest=value2 -->
<input type="checkbox" id="coding" name="checkbox" value="value1" />
<input type="checkbox" id="coding" name="checkbox" value="value2" />
```

- - color
- - date - календарь для даты
- - - max - дата yyyy-MM-dd
- - - min - дата yyyy-MM-dd
- - - step

```js
// valueAsNumber
var dateControl = document.querySelector('input[type="date"]');
dateControl.value = "2017-06-01";
console.log(dateControl.value); // prints "2017-06-01"
console.log(dateControl.valueAsNumber); // prints 1496275200000, a JavaScript timestamp (ms)
```

- - datetime - дата + время (час:минута:секунда) 00:00:30.75,
- - datetime-local - с учетом настройки ос
    данные отправляются partydate=2017-06-01T08:30
- - email (есть псевдоклассы :valid, :invalid.),
- - - multiple "me@example" "me@example.org""me@example.org,you@example.org" "me@example.org, you@example.org""me@example.org you@example.org, us@example.org"
- - - size - количество символов в ширину
- - file
- - - value - путь к файлу
- - - accept - типы обрабатываемых файлов accept="image/png, image/jpeg"или accept=".png, .jpg, .jpeg"
- - - capture - работает только на моб устройствах:
- - - - user - использовать камеру, микрофон обращенные к пользователю
- - - - environment - использовать камеру, микрофон обращенные наружу
- - hidden (элемент управления не отображается, но на сервер значение отправляется)
- - image - кнопка отправки в виде изображения, атрибуты:
- - - type
- - - formaction - переопределит action родительской формы
- - - formenctype:
- - - - application/x-www-form-urlencoded:
- - - - multipart/form-data: если type === file
- - - - text/plain
- - - formmethod - get, post
- - - formtarget
- - - height, width
- - - src
- - - usemap - если изображение часть map
- - month - YYYY-MM 1953-W01
- - number
- - password
- - - autocomplete: "on", "off", "current-password", "new-password"
- - radio

```html
<fieldset>
  <legend>Select a maintenance drone:</legend>

  <div>
    <!-- один и тот же name -->
    <!-- value - обязательно, так как отправится on на сервер при подтверждении формы -->

    <input type="radio" id="huey" name="drone" value="huey" checked />
    <label for="huey">Huey</label>
  </div>

  <div>
    <input type="radio" id="dewey" name="drone" value="dewey" />
    <label for="dewey">Dewey</label>
  </div>

  <div>
    <input type="radio" id="louie" name="drone" value="louie" />
    <label for="louie">Louie</label>
  </div>
</fieldset>
```

- - range (min, max, value, step)
- - - list - позволяет указать datalist со значениями
- - - orient(только ff): horizontal | vertical расположение

```scss
// сделать вертикальным
input[type="range"] {
  -webkit-appearance: slider-vertical;
}
```

- - reset (кнопка сброса)

```html
<input type="reset" value="Reset the form" />
```

- - search - разрывы строк автоматически удаляются, отличается от type text тем что на конце есть крестик
- - submit - атрибуты:
- - - value - текст на кнопке
- - - action - url куда отправится запрос
- - - formenctype:
- - - - application/x-www-form-urlencoded - encodeURI() строка https://mozilla.org/?x=%D1%88%D0%B5%D0%BB%D0%BB%D1%8B"
- - - - multipart/form-data - если есть input type file
- - - - text/plain
- - - method:
- - - - get
- - - - post
- - - - dialog - закроет форму диалогово окна и не отправит данные
- - - formnovalidate
- - - formtarget
- - - - \_self
- - - - \_blank
- - - - \_parent
- - - - \_top
- - tel
- - text: атрибуты
- - - list - id data-list для поставления опций

```html
<input type="email" size="40" list="defaultEmails" />

<datalist id="defaultEmails">
  <option value="jbond007@mi6.defence.gov.uk"></option>
  <option value="jbourne@unknown.net"></option>
  <option value="nfury@shield.org"></option>
  <option value="tony@starkindustries.com"></option>
  <option value="hulk@grrrrrrrr.arg"></option>
</datalist>
```

- - - maxlength
- - - minlength
- - - pattern
- - - placeholder
- - - readonly
- - - size
- - - spellcheck: true | false | ''
      :valid и :invalid псевдоэлементы

- - time - форматы HH:mm или HH:mm:ss

```html
<label for="appt">Choose a time for your meeting:</label>
<input type="time" id="appt" name="appt" min="09:00" max="18:00" required />
```

- - - datalist
- - - max
- - - min
- - - step - в секундах

- - url (есть псевдоклассы :valid, :invalid)

```html
<!-- в данном случае список будет с лейблами и url -->
<input id="myURL" name="myURL" type="url" list="defaultURLs" />

<datalist id="defaultURLs">
  <option value="https://developer.mozilla.org/" label="MDN Web Docs"></option>
  <option value="http://www.google.com/" label="Google"></option>
  <option value="http://www.microsoft.com/" label="Microsoft"></option>
  <option value="https://www.mozilla.org/" label="Mozilla"></option>
  <option value="http://w3.org/" label="W3C"></option>
</datalist>
```

- - week - формат yyyy-Www
- usemap - если часть map
- value - изначальное значение
- x-moz-errormessage - текст ошибки для Mozilla

## input type file BP. C Карточками фотографий

```html
<form method="post" enctype="multipart/form-data">
  <div>
    <label for="image_uploads">Choose images to upload (PNG, JPG)</label>
    <input
      type="file"
      id="image_uploads"
      name="image_uploads"
      accept=".jpg, .jpeg, .png"
      multiple
    />
  </div>
  <div class="preview">
    <p>No files currently selected for upload</p>
  </div>
  <div>
    <button>Submit</button>
  </div>
</form>
```

```js
// получаем элементы
var input = document.querySelector("input");
var preview = document.querySelector(".preview");

// скрываем input, по клику на label
input.style.opacity = 0;

input.addEventListener("change", updateImageDisplay);

var fileTypes = ["image/jpeg", "image/pjpeg", "image/png"];

function validFileType(file) {
  for (var i = 0; i < fileTypes.length; i++) {
    if (file.type === fileTypes[i]) {
      return true;
    }
  }

  return false;
}

function returnFileSize(number) {
  if (number < 1024) {
    return number + "bytes";
  } else if (number > 1024 && number < 1048576) {
    return (number / 1024).toFixed(1) + "KB";
  } else if (number > 1048576) {
    return (number / 1048576).toFixed(1) + "MB";
  }
}

function updateImageDisplay() {
  while (preview.firstChild) {
    preview.removeChild(preview.firstChild);
  }

  var curFiles = input.files;
  if (curFiles.length === 0) {
    var para = document.createElement("p");
    para.textContent = "No files currently selected for upload";
    preview.appendChild(para);
  } else {
    var list = document.createElement("ol");
    preview.appendChild(list);
    for (var i = 0; i < curFiles.length; i++) {
      var listItem = document.createElement("li");
      var para = document.createElement("p");
      if (validFileType(curFiles[i])) {
        para.textContent =
          "File name " +
          curFiles[i].name +
          ", file size " +
          returnFileSize(curFiles[i].size) +
          ".";
        var image = document.createElement("img");
        image.src = window.URL.createObjectURL(curFiles[i]);

        listItem.appendChild(image);
        listItem.appendChild(para);
      } else {
        para.textContent =
          "File name " +
          curFiles[i].name +
          ": Not a valid file type. Update your selection.";
        listItem.appendChild(para);
      }

      list.appendChild(listItem);
    }
  }
}
```

<!-- label ------------------------------------------------------------------------------------------------------------------->

# label

Атрибуты:

- глобальные
- for - id инпута
- form - позволяет встроить в любом месте

```html
<label for="input">
  имя
  <label>
    - при клике на label фокус переход на inputс таким же id</label
  ></label
>
```

<!-- menu ----------------------------------------------------------------------------------------------------------------------->

# menu

для отображающегося меню

Атрибуты:

- label
- type:
- - context: для нажатия пкм (не работает)
- - toolbar - тогда тег menu должен быть внутри li

<!-- meter ------------------------------------------------------------------------------------------------------------------->

# meter (HTML5)

Покажет шкалу с цветовым определением

```html
<p>
  Heat the oven to <meter min="200" max="500" value="350">350 degrees</meter>.
</p>
```

Атрибуты:

- value - текущее значение
- min
- max
- low - меньше этого значения будет один цвет
- high - больше этого значения будет один цвет
- optimum
- form - для связи с формой

<!-- output ------------------------------------------------------------------------------------------------------------------->

# output (HTML5)

Контейнер вывода информации

Атрибуты:

- for - одно значение id или через запятую, которое нужно вывести
- form - нужен id формы, если элемент не вложен в form
- name - нужен при отправки формы

```html
<form oninput="result.value=parseInt(a.value)+parseInt(b.value)">
  <input type="range" name="b" value="50" /> +
  <input type="number" name="a" value="10" /> =
  <output name="result">60</output>
</form>
```

<!-- progress ------------------------------------------------------------------------------------------------------------------->

# progress (HTML5)

Прогресс бар

Атрибуты:

- max
- value

CSS псевдоклассы:

- -moz-orient
- :indeterminate
<!-- search ------------------------------------------------------------------------------------------------------------------->

# search

тег-обертка для поиска и фильтрации содержимого

```html
<header>
  <h1>Movie website</h1>
  <search>
    <form action="./search/">
      <label for="movie">Find a Movie</label>
      <input type="search" id="movie" name="q" />
      <button type="submit">Search</button>
    </form>
  </search>
</header>
```

<!-- select option optgroup ------------------------------------------------------------------------------------------------------------------->

# select option optgroup

## select

Меню опций

Атрибуты:

- autofocus
- disabled
- form - id формы, к которой привязан
- multiple
- name - для имя
- required
- size

```html
<select name="select">
  <!--Supplement an id here instead of using 'name'-->
  <option value="value1">Значение 1</option>
  <option value="value2" selected>Значение 2</option>
  <option value="value3">Значение 3</option>
</select>
```

## option

Элемент списка select

Атрибуты:

- глобальные
- disabled
- label
- selected
- value - если нет, то берется из текста

## optgroup

позволяет группировать опции option

Атрибуты:

- глобальные
- disabled
- label - имя отображаемой группы
- value - значение

```html
<select>
  <optgroup label="Группа 1">
    <option>Опция 1.1</option>
  </optgroup>
  <optgroup label="Группа 2">
    <option>Опция 2.1</option>
    <option>Опция 2.2</option>
  </optgroup>
  <!-- вся группа будет неактивна -->
  <optgroup label="Группа 3" disabled>
    <option>Опция 3.1</option>
    <option>Опция 3.2</option>
    <option>Опция 3.3</option>
  </optgroup>
</select>
```

<!-- textarea ------------------------------------------------------------------------------------------------------------------>

# textarea

Поле textarea представляет собой элемент формы для создания области, в которую можно вводить несколько строк текста. В отличие от тега input в текстовом поле допустимо делать переносы строк, они сохраняются при отправке данных на сервер.
Между тегами textarea и /textarea можно поместить любой текст, который будет отображаться внутри поля.

inline-block

Атрибут:

- autocapitalize
- autocomplete: off, on
- autocorrect: on, off
- autofocus
- cols - ширина в примерных размерах символов, по умолчанию === 20
- dirname - направленность текста
- disabled
- form
- maxlength
- minlength
- name
- placeholder
- readonly
- required
- rows
- spellcheckЖ true, false
- wrap - как обработать значение при отправке
- - hard,
- - soft
- - off

```scss
textarea {
  resize: none; //
  resize: vertical;
  field-sizing: content;
}
```
