<!-- a ----------------------------------------------------------------------------------------------------------------------->

# a (str)

!!!Ссылки являются встроенным элементом, внутри тега a нельзя располагать блочные элементы и наоборот вкладывать ссылку в блочный контейнер

аттрибуты:

- href единственный обязательный атрибут. mail.to Создание
  ссылки на электронный адрес электронной почты с атрибутом mailto:адрес эл почты,
  при нажатии запускается почтовая программа, можно добавить параметр через ?
  subject = тема сообщения

```html
<a href="mailto:name@mail.ru?subject=Тема письма">Задавайте вопросы</a>

<a
  href="mailto:nowhere@mozilla.org?cc=name2@rapidtables.com&bcc=name3@rapidtables.com&amp;subject=The%20subject%20of%20the%20email &amp;body=The%20body%20of%20the%20email"
>
  Отправить письмо с полями cc, bcc, subject и body
</a>
```

Также параметрами строки могут выступать «subject», «cc» и «body»

- download - если есть значение у этого атрибута, то файл будет скачен с таким именем
- hreflang - язык документа по ссылке
- ping - уведомляет указанные в нём URL, что пользователь перешёл по ссылке
- referrerpolicy какую информацию передавать по ссылке
- - no-referrer - без заголовка Referer
- - no-referrer-when-downgrade - не отправляет заголовок Referer ресурсу без TLS HTTPS
- - origin - отправит информацию о странице адрес итд
- - origin-when-cross-origin - путь отправит только внутри ресурса
- - unsafe-url - отправляет только ресурс и адрес
- rel - устанавливает отношения между ссылками
- target - по умолчанию открывается в текущем окне или фрейме, но можно изменить с помощью target

```html
<a target="Имя окна">…</a>

<!-- в качестве значения используется имя окна или фрейма. -->
<!-- Зарезервированные имена:  -->
<!-- _blank – загружает страницу в новое окно браузера-->
<a target="_blank"></a>
<!-- загружает страницу в текущее окно  -->
<a target="_self"></a>
<!--_parent загружает страницу во фрейм-родитель-->
<a target="_parent"></a>
<!--_top отменяет все фреймы-->
<a target="_top"></a>
<!--для работы с фреймами-->
<a target="_unfencedTop"></a>
```

- title - дополнительная информация о ссылке (будет отображаться)

```html
<a href="link" title="дополнительная информация при наведении"></a>
```

- type - определяет MIME тип

- устаревшие: charset, coords, name, rev, shape
- нестандартные: datafld, datasrc, methods, urn

## Якоря

Якорь – закладка с уникальным именем на определенном месте страницы, для создания перехода к ней

```html
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html" charset="utf-8" />
    <title>Быстрый доступ внутри документа</title>
  </head>
  <body>
    <p><a name="top"></a></p>
    <p>…</p>
    <p><a href="#top">Наверх </a></p>
    <p></p>
  </body>
</html>
```

в общем виде

```html
<div id="id">Объект навигации</div>

<a href="document-name.html#id">наведет на div выше</a>
```

## BP. Создание кликабельной картинки

```html
<a href="https://developer.mozilla.org/ru/" target="_blank">
  <img src="mdn_logo.png" alt="MDN logo" />
</a>
```

## BP. Создание ссылки с номером телефона

```html
<a href="tel:+491570156">+49 157 0156</a>
```

## BP. сохранение рисунка canvas

```js
var link = document.createElement("a");
link.innerHTML = "download image";

link.addEventListener(
  "click",
  function (ev) {
    link.href = canvas.toDataURL();
    link.download = "myPainting.png";
  },
  false
);

document.body.appendChild(link);
```

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

<!-- details ----------------------------------------------------------------------------------------------------------------->

# details и summary (block, HTML5)

Раскрывающееся меню вниз

[summary - тег будет использован как заголовок](#summary-html5)

```html
<details>
  <summary>Details</summary>
  Something small enough to escape casual notice.
</details>
```

Атрибуты:

- open - изначальное состояние

стилизовать маркер можно с помощью ::-webkit-details-marker
summary {display: block;} для того что бы скрыть треугольник по умолчанию и добавить свой

Добавление собственного элемента для summary

```html
<details>
  <summary>Some details</summary>
  <p>More info about the details.</p>
</details>
```

```scss
summary {
  display: block;
}

summary::-webkit-details-marker {
  display: none;
}

summary::before {
  content: "\25B6";
  padding-right: 0.5em;
}

details[open] > summary::before {
  content: "\25BC";
}
```

<!-- dialog ----------------------------------------------------------------------------------------------------------------->

# dialog (block) (block, HTML5)

Элемент диалогового окна. Пример с окном выбора email. нельзя присваивать tabIndex
Атрибуты:

- open

::backdrop - позволяет стилизовать подложку

```html
<!-- Простой попап диалог с формой -->
<dialog id="favDialog">
  <form method="dialog">
    <section>
      <p>
        <label for="favAnimal">Favorite animal:</label>
        <select id="favAnimal">
          <option></option>
          <option>Brine shrimp</option>
          <option>Red panda</option>
          <option>Spider monkey</option>
        </select>
      </p>
    </section>
    <menu>
      <button id="cancel" type="reset">Cancel</button>
      <button type="submit">Confirm</button>
    </menu>
  </form>
</dialog>

<menu>
  <button id="updateDetails">Update details</button>
</menu>

<script>
  (function () {
    var updateButton = document.getElementById("updateDetails");
    var cancelButton = document.getElementById("cancel");
    var favDialog = document.getElementById("favDialog");

    // Update button opens a modal dialog
    updateButton.addEventListener("click", function () {
      favDialog.showModal();
    });

    // Form cancel button closes the dialog box
    cancelButton.addEventListener("click", function () {
      favDialog.close();
    });
  })();
</script>
```

<!-- fieldset legend----------------------------------------------------------------------------------------------------------->

# fieldset (block) legend

применяется для создания заголовка группы элементов формы, которая определяется с помощью тега fieldset. Группа элементов обозначается в браузере с помощью рамки, а текст, который располагается внутри контейнера legend, встраивается в эту рамку.

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

- accept - какие файлы может принимать input
- - "audio/", "video/", "image/"", "image/png", ".png"
- accesskey - управление фокусом
- mozactionhint - определяет кнопку на моб телефонах go, done, next, search, и send
- autocomplete: "off", "on" или какая-либо строка через пробел
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
- usemap
- value - изначальное значение
- x-moz-errormessage - текст ошибки для Mozilla

- type: text, button, checkbox, color, date, datetime, datetime-local, email (есть псевдоклассы :valid, :invalid.), file, hidden (элемент управления не отображается, но на сервер значение отправляется), image, month, number, password, radio, range (min, max, value, step), reset (кнопка сброса), search (разрывы строк автоматически удаляются), submit, tel, text, time, url (есть псевдоклассы :valid, :invalid.), week

## input type file

атрибуты:

- capture - работает только на моб устройствах:
- - user - использовать камеру, микрофон обращенные к пользователю
- - environment - использовать камеру, микрофон обращенные наружу

<!-- label ------------------------------------------------------------------------------------------------------------------->

# label

Атрибуты:

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

<!-- search ------------------------------------------------------------------------------------------------------------------->

# search

тег для поиска и фильтрации

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

- disabled
- label
- selected
- value - если нет, то берется из текста

## optgroup

позволяет группировать опции option

Атрибуты:

- disabled
- label - имя отображаемой группы

```html
<select>
  <optgroup label="Группа 1">
    <option>Опция 1.1</option>
  </optgroup>
  <optgroup label="Группа 2">
    <option>Опция 2.1</option>
    <option>Опция 2.2</option>
  </optgroup>
  <optgroup label="Группа 3" disabled>
    <option>Опция 3.1</option>
    <option>Опция 3.2</option>
    <option>Опция 3.3</option>
  </optgroup>
</select>
```

<!-- summary ------------------------------------------------------------------------------------------------------------------->

# summary (HTML5)

Видимы заголовок для [details](#details-и-summary-block-html5)

display: list-item

<!-- textarea ------------------------------------------------------------------------------------------------------------------>

# textarea

Поле textarea представляет собой элемент формы для создания области, в которую можно вводить несколько строк текста. В отличие от тега input в текстовом поле допустимо делать переносы строк, они сохраняются при отправке данных на сервер.
Между тегами textarea и /textarea можно поместить любой текст, который будет отображаться внутри поля.

Атрибут:

- autocapitalize
- autocomplete: off, on
- autocorrect: on, off
- autofocus
- cols
- dirname
- disabled
- form
- maxlength
- minlength
- name
- placeholder
- readonly
- required
- rows
- spellcheck
- wrap