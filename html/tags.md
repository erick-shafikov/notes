!!!TODO tr, th, tfoot из mdn

- (str) - строчный элемент
- (block) - блочный
- (HTML5) - семантический тег, каждый такой тег начинает с определения структурны он или поточный

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
```

Также параметрами строки могут выступать «subject», «cc» и «body»

- download - если есть значение у этого атрибута, то файл будет скачен с таким именем
- hreflang - язык документа по ссылке
- ping - уведомляет указанные в нём URL, что пользователь перешёл по ссылке
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

- referrerpolicy: no-referrer, no-referrer-when-downgrade, origin, origin-when-cross-origin, "unsafe-url
- rel (глобальный атрибут)
- title

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
    <meta http-equiv=Content-Type content =text/html; charset = utf-8>
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

## сохранение рисунка canvas

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

<!-- abbr ----------------------------------------------------------------------------------------------------------------------->

# abbr (inline)

для аббревиатур, атрибут title покажет полную расшифровку

Может быть использован в паре с тегом [dfn](#dfn-str)

```html
<p>
  <dfn id="html"><abbr title="HyperText Markup Language">HTML</abbr> </dfn> –
  язык разметки...
</p>

<p>
  A <dfn id="spec">Specification</dfn> (<abbr title="Specification">spec</abbr>)
  – документ...
</p>
```

Стал заменой тегу acronym

<!-- address ------------------------------------------------------------------------------------------------------------------->

# address (block)

задает контакты для ближайшего родительского article или body. Для контактной информации использую тег p, нельзя вкладывать time, обычно внутри footer

<!-- article ----------------------------------------------------------------------------------------------------------------------->

# article (block, HTML5)

Структурный тег. создан для отдельной смысловой единице, которую можно оторвать от сайта. Размер текста 1.5 rem === 24px

!!! должен быть идентифицирован добавляя теги h1-h6
!!! может быть вложен

<!-- aside ----------------------------------------------------------------------------------------------------------------------->

# aside (block, HTML5)

Структурный тег. для неосновного контента. отступление предназначен для отступления. например пометки на полях в печатном
журнале

<!-- audio ------------------------------------------------------------------------------------------------------------------->

# audio (block, HTML5)

теги:

- track - для файла с субтитрами в формате vtt
- source - для файлов, будет по очереди стараться загрузить нужный

атрибуты:

- autoplay - автоматическое воспроизведение
- controls - добавит элементы управления аудио, если не указан, то элементов управления не будет на экране
- crossorigin:
- - anonymous - без передачи cookie
- - use-credentials
- loop
- muted
- preload
- - none
- - metadata
- - auto
- src - можно использовать вместо source

```html
<audio controls>
  <source src="viper.mp3" type="audio/mp3" />
  <source src="viper.ogg" type="audio/ogg" />
  <track kind="subtitles" src="subtitles_en.vtt" srclang="en" />
  <p>
    Your browser doesn't support HTML5 audio. Here is a
    <a href="viper.mp3">link to the audio</a> instead.
  </p>
</audio>
```

если нет атрибута control то визуальных элементов не будет, если задан, то будет отображаться обычный inline элемент, который позволяет стилизовать только контейнер, но не элементы управления

<!-- b ----------------------------------------------------------------------------------------------------------------------->

# b (str)

Жирный шрифт, по важности уступает strong, лучше использовать font-weight

Атрибуты:

- href
- target: \_self, \_blank, \_parent, \_top

<!-- base ----------------------------------------------------------------------------------------------------------------------->

# base

Определяет основной url страницы, родители - head, body

<!-- bdi --------------------------------------------------------------------------------------------------------------------->

# bdi, bdo (block, HTML5)

bdi Изолирует двунаправленное определение текста

```html
<p dir="ltr">
  This arabic word <bdi>ARABIC_PLACEHOLDER</bdi> is automatically displayed
  right-to-left.
</p>
```

bdo поддерживает атрибут dir ltr и rtl

<!-- blockquote ----------------------------------------------------------------------------------------------------------------->

# blockquote (block)

Если идет цитируемость с другого источника и нужно заключить в новый абзац, если нужна строчная цитата [цитаты строчная](#q)

```html
<blockquote cite="href-откуда-цитата">
  <p>текст цитаты</p>
</blockquote>
```

<!-- body --------------------------------------------------------------------------------------------------------------->

# body

Атрибуты:

- background фоновое изображение (лучше через css)
- bgcolor - цвет фона (лучше через css)
- цвет ссылок:
- - alink - цвет текста гиперссылок (лучше через css)
- - link - цвет непосещенных гиперссылок (лучше через css)
- - vlink - цвет посещенной ссылки (лучше через css)
- margin:
- - bottommargin - отступ внизу (лучше через css)
- - leftmargin - отступ слева (лучше через css)
- - rightmargin - отступ справа (лучше через css)
- - topmargin
- коллбеки:
- - onafterprint - ф-ция будет вызвана после печати
- - onbeforeprint - до печати
- - onbeforeunload - перед закрытием окна
- - onblur - при потере фокуса
- - onfocus - при фокусировки
- - onhashchange - при изменении части идентификатора #
- - onlanguagechange - при смене языка
- - onload - при загрузке страницы
- - onoffline - при потере соединения
- - ononline - при восстановлении соединения
- - onpopstate - изменение истории
- - onredo - при движении в перед по истории
- - onundo - при движении назад по истории
- - onresize - при изменении размера
- - onstorage- при изменении в хранилище
- - onunload - при закрытии окна браузера

<!-- br --------------------------------------------------------------------------------------------------------------------->

# br

создает разрыв строки, нет стилей, но можно использовать margin, а лучше line-height

<!-- button --------------------------------------------------------------------------------------------------------------------->

# button

Допустимое содержимое - текстовый контент

Атрибуты:

- autofocus
- disabled
- form - содержит id формы
- formaction - ссылка на обработчик формы
- formenctype:
- - application/x-www-form-urlencoded - по умолчанию
- - multipart/form-data - если type file у input
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

<!-- canvas  ----------------------------------------------------------------------------------------------->

# canvas (block, HTML5)

Область для отрисовки

Атрибуты:

- height - 150 по умолчанию
- width - 300 по умолчанию
- moz-opaque - полупрозрачность

<!-- cite  ----------------------------------------------------------------------------------------------->

# cite

ссылка на какой-либо источник

<!-- code, pre, var, kbd, samp  ----------------------------------------------------------------------------------------------->

# code, pre(block), var, kbd, samp

- code - для кода
- pre - для кода, в котором ненужно игнорировать пробелы, в него нужно обернуть code, если более одной строки кода в ставке или текст в котором нужно соблюсти пробелы и разрывы строк
- var - для маркировки переменных
- kbd - для маркировки ввода с клавиатуры
- samp - для маркировки ввод кода

<!-- data ----------------------------------------------------------------------------------------------------------------->

# data (block, HTML5)

добавит машиночитаемый код

```html
<ul>
  <!-- каждый элемент связан со своим id -->
  <li><data value="398">Mini Ketchup</data></li>
  <li><data value="399">Jumbo Ketchup</data></li>
  <li><data value="400">Mega Jumbo Ketchup</data></li>
</ul>
```

<!-- datalist ----------------------------------------------------------------------------------------------------------------->

# datalist (block, HTML5)

отображает всплывающий tooltip с опциями, непосредственно связан с input по id

```html
<label for="ice-cream-choice">Choose a flavor:</label>
<input list="ice-cream-flavors" id="ice-cream-choice" name="ice-cream-choice" />

<datalist id="ice-cream-flavors">
  <option value="Chocolate"></option>
  <option value="Coconut"></option>
  <option value="Mint"></option>
  <option value="Strawberry"></option>
  <option value="Vanilla"></option>
</datalist>
```

<!-- del ins ---------------------------------------------------------------------------------------------------------------------->

# del ins

del Отобразит перечеркнутый текст, ins - добавленный текст

Атрибуты:

- cite - урл причины удаления
- datetime - дата удаления

<!-- details ----------------------------------------------------------------------------------------------------------------->

# details и summary (block, HTML5)

Раскрывающееся меню вниз

```html
<details>
  <summary>Details</summary>
  Something small enough to escape casual notice.
</details>
```

Атрибуты:

- open - изначальное состояние

стилизовать маркер можно с помощью ::-webkit-details-marker

<!-- del, ins ----------------------------------------------------------------------------------------------------------------->

# del (str)

<!-- dfn ----------------------------------------------------------------------------------------------------------------->

# dfn (str)

используется в паре с [abbr](#abbr-inline)

<!-- dialog ----------------------------------------------------------------------------------------------------------------->

# dialog (block) (block, HTML5)

Элемент диалогового окна. Пример с окном выбора emil. нельзя присваивать tabIndex

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

<!-- div ----------------------------------------------------------------------------------------------------------------->

# div (block)

<!-- dl, dt, dd ----------------------------------------------------------------------------------------------------------------->

# dl, dt, dd (block)

Каждый такой список начинается с контейнера dl(description list), куда входит тег dt создающий термин и тег dd (description details) задающий определение этого термина. Закрывающий тег dt не обязателен, поскольку следующий тег сообщает о завершении предыдущего элемента. Тем не менее, хорошим стилем является закрывать все теги.

dl Имеет вертикальные по 16px
dd имеет margin-left === 2.5rem

```html
Синтаксис
<dl>
  <dt>Термин 1</dt>
  <dd>Определение термина 1</dd>
  <dt>Термин 2</dt>
  <dd>Определение термина 2</dd>
</dl>
```

<!-- em ---------------------------------------------------------------------------------------------------------------------->

# em (str)

Отмечает акцентированный текст, отображается курсивом. Разница с i

<!-- embed ---------------------------------------------------------------------------------------------------------------------->

# embed (block, HTML5)

устаревший вариант для встраивания контента, атрибуты: height, src, type, width

<!-- figure figcaption----------------------------------------------------------------------------------------------------------->

# figure и figcaption (block, HTML5)

Потоковый элемент. Тег картинки и подписи к ней. теги нужны для улучшения семантики

```html
<figure class="story__shape">
  <img src="img/nat-8.jpg" alt="person on a tour" class="story__img" />
  <figcaption class="story__caption">Mary Smith</figcaption>
</figure>
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

<!-- footer----------------------------------------------------------------------------------------------------------->

# footer (block, HTML5)

для контента внизу страницы, нижний колонтитул, на количество ограничения не накладываются

<!-- form ---------------------------------------------------------------------------------------------------------------------->

# form (block)

Атрибуты:

- accept-charset - кодировки на сервер
- action - урл куда отправлять данные формы. В качестве обработчика может выступать CGI-программа или HTML-документ, который включает в себя серверные сценарии (например, Parser). После выполнения обработчиком действий по работе с данными формы он возвращает новый HTML-документ.
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

<!-- h1-h6 ---------------------------------------------------------------------------------------------------------------------->

# h1-h6 (block)

Один h1 на страниц. h1 === 2rem (16px)

<!-- head ---------------------------------------------------------------------------------------------------------------------->

# head

Не отображается в документе, главная цель – метаданные, содержит:

title - отображает заголовок на странице

Внутри себя использует:

- [тег link](#link)
- [теги meta](#meta)
- [script](#script)

Создается автоматически

<!-- header ---------------------------------------------------------------------------------------------------------------------->

# header (block, HTML5)

Потоковый тег. Если это дочерний элемент body, то это заголовок всей страницы, также может быть заголовком section или article

<!--  hgroup ---------------------------------------------------------------------------------------------------------------->

# hgroup (block, HTML5)

Группирует h1-h6 в один заголовок или группирует несколько тегов p

<!-- hr ------------------------------------------------------------------------------------------------------------------>

# hr (block)

горизонтальная черта. Устаревшие атрибуты: align, color, noshade, size, width

<!-- html ---------------------------------------------------------------------------------------------------------------------->

# html

Корневой элемент для все страницы. В нем должен быть head и body

Атрибуты:

- manifest - uri манифеста
- xmlns

```html
<html lang="en"></html>
```

выделяет символы

<!-- i ---------------------------------------------------------------------------------------------------------------------->

# i

Курсив

<!-- iframe ------------------------------------------------------------------------------------------------------------------->

# iframe

Нужен для отображения другой страницы в контексте текущей

атрибуты:

- allowfullscreen - возможность открыть фрейм в полноэкранном режиме
- frameborder - обозначить границу, значения 0 и 1, лучше использовать border
- loading: eager, lazy
- name - для фокусировки
- src
- width, height
- sandbox - повышает настройки безопасности

```html
<iframe
  src="https://developer.mozilla.org/ru/docs/Glossary"
  width="100%"
  height="500"
  frameborder="0"
  allowfullscreen
  sandbox
>
  <p>fallback</p>
</iframe>
```

Фреймы – разделяют окно браузера на отдельные области расположенные вплотную друг у другу, в каждый загружается отдельная веб страница. Позволяют открыть документ в одном фрейме по ссылке нажатой в совершенно в другом фрейме. поддерживают вложенную структуру

frame определяет свойство отдельного фрейма
frameset заменяет элемент body на веб странице и формирует структуру фреймов (deprecated)
iframe создает плавающий фрейм, который находится внутри обычного документа

```html

<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Frameset//EN" "http://www.w3.org/TR/html4/frameset.dtd">
<!--поле для использования фреймов-->
<html>
  <head>
    <meta />
    <title>Фреймы</title>
  </head>
  <frameset cols="100">
    <!-- //левая колонка 100px правая все оставшиеся пространство -->
    <iframe src="menu.html" name="MENU">
      <!-- //должно быть 2 файла -->
      <iframe src="content.html" name="CONTENT">
      </iframe></iframe></frameset>
    <!-- //атрибут rows создал 2 горизонтальных фрейма -->
  <frameset rows="25%, 75%">

    <frame src="top.html" name="top" scrolling="no" noresize>
    <frameset cols="100">
      <frame src="menu.html" name="MENU"
      <frame src="content.html" name="CONTENT">
     </frameset>
  </frameset>

</html>
```

## Ссылки во фреймах

для загрузки фреймов один в другой используется атрибут target тега a

```html
<frameset cols="100, *">
  <frame src="menu2.html" name="MENU">
  <frame src="content.html" name="CONTENT">
</frameset>

<!-- menu2: -->
<body>
  <p>МЕНЮ</p>
  <p><a href="text.html" target="CONTENT"> ТЕКСТ </a></p>
</body>
```

имя фрейма должно начинаться на цифру или латинскую букву в качестве зарезервированных имен

\_blank –загружает документ в новое коно
\_self – загружает документ в новый фрейм
\_parent – загружает документ во фрейм, занимаемый родителем если фрейма-родителя нет значения действет также как \_top
\_top – отменяет все фреймы и загружает документ в полное окно браузер

## границы, размер, прокрутка

Убираем границу между фреймами

```html
<frameset cols="100,*" frameborder="0" framesapcing="0">
  <bordercolor =""> – атрибут для смены цвета рамки </bordercolor></frameset
>
```

Изменение размеров

для блокировки возможности измения размера атрибут noresize

для полос прокрутки – атрибут scrolling, который принимает два значения no или yes

## Плавающие фреймы

Создание плавающего фрейма

и обязательный атрибут src

```html
<iframe>
  <p><iframe src="hsb.html" width="300" height="120"></iframe></p>

  для того чтобы загрузить документ по ссылке
  <p>
    <iframe src="model.html" name="color" width="100%" height="200"></iframe>
  </p>
</iframe>
```

<!-- img ---------------------------------------------------------------------------------------------------------------------->

# img (str)

Атрибуты:

- srcset - позволяет загружать картинки в зависимости от ширины экрана. Формат - ссылка, запятая пробел, где w - это ширина в пикселях
- sizes - медиа выражения и слот в каждой строке, соответствие с srcset по принципу самый первый, который больше

  для более тонких настроек:

- crossorigin значения:
- - anonymous
- - use-credentials
- decoding - поведение декодирования:
- - sync - синхронно с другим контентом
- - async - параллельно, что бы уменьшить задержку с другим контентом
- - auto
- importance - приоритет загрузки auto, low, high
- ismap - карта ссылок
- loading: eager, lazy
- referrerpolicy: no-referrer, no-referrer-when-downgrade, origin, origin-when-cross-origin, unsafe-url
- sizes - одна или несколько строк разделенные запятыми, состоящие из медиа запроса, размер источника
- title - лучше figure и figcaption
- usemap

```html
<img
  srcset="
    elva-fairy-320w.jpg 320w,
    elva-fairy-480w.jpg 480w 1.5x,
    elva-fairy-800w.jpg 800w 2x
  "
  sizes="(max-width: 320px) 280px,
         (max-width: 480px) 440px,
         800px
  "
  alt="photo 3"
  class="composition__photo composition__photo--p3"
  src="img/nat-3-large.jpg"
  width="100"
  height="100"
  title="заголовок изображения, всплывающая подсказка"
/>
```

для связи заголовка и изображения [figure и figcaption](#figure-и-figcaption)
для более гибкого адаптивного поведения [picture](#picture)

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

<!-- li ----------------------------------------------------------------------------------------------------------------------->

# li (block)

элемент для списков ol ul

Атрибуты:

- value - числовой атрибут порядкового номера если это ol, с которого начнется
- type - заменен на css свойство list-style (a - алф, A - АЛФ, i - рим, I- РИМ, 1 - числа)

<!-- link ----------------------------------------------------------------------------------------------------------------------->

# link

Показывает отношение сайта и внешних ссылок

```html
<!-- основные виды применения: -->
<!-- добавление иконки -->
<link
  rel="shortcut icon"
  href="favicon.ico"
  type="image/x-icon"
  size="100x100"
/>
<!-- Подключение стилей, которая позволяет подгружать условно media="screen and (max-width: 600px)" то есть для пк и mw=600-->
<link
  rel="stylesheet"
  href="my-css-file.css"
  media="screen and (max-width: 600px)"
  title="style fro 600px"
/>
<!-- Подключение шрифтов -->
<link
  rel="preload"
  href="myFont.woff2"
  as="font"
  type="font/woff2"
  crossorigin="anonymous"
/>
```

```html
<!-- добавление иконки для разных устройств -->
<link
  rel="apple-touch-icon-precomposed"
  sizes="144x144"
  href="https://developer.mozilla.org/static/img/favicon144.png"
/>
<!-- Для iPhone с Retina-экраном высокого разрешения: -->
<link
  rel="apple-touch-icon-precomposed"
  sizes="114x114"
  href="https://developer.mozilla.org/static/img/favicon114.png"
/>
<!-- Для iPad первого и второго поколения: -->
<link
  rel="apple-touch-icon-precomposed"
  sizes="72x72"
  href="https://developer.mozilla.org/static/img/favicon72.png"
/>
<!-- Для iPhone, iPod Touch без Retina и устройств с Android 2.1+: -->
<link
  rel="apple-touch-icon-precomposed"
  href="https://developer.mozilla.org/static/img/favicon57.png"
/>
<!-- Для других случаев - обычный favicon -->
<link
  rel="shortcut icon"
  href="https://developer.mozilla.org/static/img/favicon32.png"
/>
```

Атрибуты:

- as - требуется только rel="preload" или rel="prefetch" позволяет установить приоритет
- crossorigin: anonymous, use-credentials
- href - url ресурса
- hreflang - язык
- importance: auto, high, low используется только rel="preload" или rel="prefetch"
- media - запрос для ресурса используется для внешних стилей
- referrerpolicy - no-referrer, no-referrer-when-downgrade, origin, origin-when-cross-origin, unsafe-url
- rel - определяет отношение ресурса и внешней ссылки
- sizes - только для иконок, только при rel=icon
- title - MIME тип

## Отследить загрузку стилей

```html
<script>
  var myStylesheet = document.querySelector("#my-stylesheet");

  myStylesheet.onload = function () {
    // Do something interesting; the sheet has been loaded
  };

  myStylesheet.onerror = function () {
    console.log("An error occurred loading the stylesheet!");
  };
</script>

<link rel="stylesheet" href="mystylesheet.css" id="my-stylesheet" />
```

## ref=preload

Позволяет подгрузить ресурс заранее

```html
<head>
  <meta charset="utf-8" />
  <title>JS and CSS preload example</title>

  <link rel="preload" href="style.css" as="style" />
  <link rel="preload" href="main.js" as="script" />

  <link rel="stylesheet" href="style.css" />
</head>

<body>
  <h1>bouncing balls</h1>
  <canvas></canvas>

  <script src="main.js" defer></script>
</body>
```

Предварительно могут быть загружены: fetch, font, image, script, style, track

<!-- main ----------------------------------------------------------------------------------------------------------------------->

# main (block, HTML5)

Потоковый тег. один на всю страницу, должен быть внутри body, при добавлении id позволяет упростить навигацию для устройств со спец возможностями

<!-- map и area--------------------------------------------------------------------------------------------------------------->

# map и area

используется для интерактивной областью ссылки

```html
<map name="infographic">
  <area
    shape="poly"
    coords="130,147,200,107,254,219,130,228"
    href="https://developer.mozilla.org/docs/Web/HTML"
    target="_blank"
    alt="HTML"
  />
  <area
    shape="poly"
    coords="130,147,130,228,6,219,59,107"
    href="https://developer.mozilla.org/docs/Web/CSS"
    target="_blank"
    alt="CSS"
  />
  <area
    shape="poly"
    coords="130,147,200,107,130,4,59,107"
    href="https://developer.mozilla.org/docs/Web/JavaScript"
    target="_blank"
    alt="JavaScript"
  />
</map>
<img
  usemap="#infographic"
  src="/media/examples/mdn-info2.png"
  alt="MDN infographic"
/>
```

Атрибуты area:

- accesskey - позволяет перейти к элементу по нажатию клавиатуры
- alt - для альтернативного текста, если изображение не прогрузилось
- coords - активная область
- download - если ссылка для скачивания файла
- href - ссылка
- hreflang
- name - должен совпадать с id если есть
- media: print, screen, all
- nohref
- referrerpolicy: no-referrer, no-referrer-when-downgrade, origin, origin-when-cross-origin, unsafe-url
- rel
- shape
- tabindex
- target: \_self, \_blank, \_parent, \_top
- type - mime type

```html
<map name="primary">
  <area shape="circle" coords="75,75,75" href="left.html" />
  <area shape="circle" coords="275,75,75" href="right.html" />
</map>
<img usemap="#primary" src="https://placehold.it/350x150" alt="350 x 150 pic" />
```

Атрибуты area:

- name

<!-- mark ----------------------------------------------------------------------------------------------------------------------->

# mark (HTML5)

Предназначен для выделение текста в результате поиска. Отличие от strong - mark используется для отношения к другому контексту

<!-- marquee ----------------------------------------------------------------------------------------------------------------------->

# marquee

для вращающегося текста

<!-- menu ----------------------------------------------------------------------------------------------------------------------->

# menu

для отображающегося меню

Атрибуты:

- label
- type:
- - context: для нажатия пкм (не работает)
- - toolbar - тогда тег menu должен быть внутри li

<!-- meta ----------------------------------------------------------------------------------------------------------------------->

# meta

Синтаксис - атрибуты name и content
Существую og метаданные open graph для facebook, так же есть у твиттера

```html
<!-- кодировка страницы -->
<meta charset="utf-8" />
<!-- Автор -->
<meta name="author" content="автор" />
<!-- Описание, то что будет видно в поисковой выдачи -->
<meta name="description" content="описание" />
<!--  -->

<!-- метаданные open graph -->
<meta property="og:image" content="__href-to-image__" />
<meta property="og:description" content="описание" />
<meta property="og:title" content="титул" />
```

Атрибуты:

- charset
- content - определяет значение для атрибутов http-equiv или name
- http-equiv значения значения для content:
- - "content-language" - язык страницы
- - "Content-Security-Policy" -
- - "content-type" - MIME type документа
- - "default-style" - content атрибут должен содержать заголовок link элемента который href
- - "refresh" - Количество секунд перезагрузки таблицы или время через запятую и ресурс перенаправления
- - "set-cookie"

```html
<meta http-equiv="refresh" content="3;url=https://www.mozilla.org" />
```

- name - не следует указывать, если установлены itemprop, http-equiv или charset
- - application-name
- - referrer
- - creator
- - googlebot
- - publisher
- - robots
- - scheme

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

<!-- nav ---------------------------------------------------------------------------------------------------------------------->

# nav (block, HTML5)

Структурный тег. для навигации по сайту. используется для навигационных ссылок. Док может содержать несколько nav

```html
<nav class="menu">
  <ul>
    <li><a href="#">Главная</a></li>
    <li><a href="#">О нас</a></li>
    <li><a href="#">Контакты</a></li>
  </ul>
</nav>
```

<!-- noscript ------------------------------------------------------------------------------------------------------------------->

# noscript

дочерние элементы этого тега будут отображаться есть нет поддержки js

<!-- object ------------------------------------------------------------------------------------------------------------------->

# object

предназначен для встраивания контента, если он может быть интерпретирован как изображение

```html
<object
  data="mypdf.pdf"
  type="application/pdf"
  width="800"
  height="1200"
  typemustmatch
>
  <p>
    You don't have a PDF plugin, but you can
    <a href="mypdf.pdf">download the PDF file.</a>
  </p>
</object>
```

Атрибуты:

- data
- form
- height
- name
- type
- width

<!-- ol ------------------------------------------------------------------------------------------------------------------->

# ol (block)

имеет верхний и нижний margin по 16px === 1em, padding-left === 1.5 em (40px)

ordered list

```html
<ol>
  <li>Первый пункт</li>
  <li>Первый пункт</li>
</ol>
```

```html
Атрибут type для выбора типа маркеров:
<!-- Арабские числа  -->
<ol type="1"></ol>
<!-- Прописные буквы  -->
<ol type="A"></ol>
<!-- Строчные буквы  -->
<ol type="a"></ol>
<!-- римские числа в верхнем регистре  -->
<ol type="I"></ol>
<!-- римские числа в нижнем регистре  -->
<ol type="i"></ol>
```

Атрибут start для начала списка с определенного значения

```html
<ol type="I" start="8"></ol>
```

Атрибуты:

- reversed
- start - число с которого начинается нумерация
- type - заменен на css свойство list-style (a - алф, A - АЛФ, i - рим, I- РИМ, 1 - числа)

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

<!-- p ------------------------------------------------------------------------------------------------------------------->

# p (block)

HTML не устанавливает перенос текста

абзац – необязательный тег.
Выравнивание с помощью атрибута align, также выравнивать можно с помощью тега div. align может принимать:

- left – выравнивание по левому краю (по умолчанию)
- right – выравнивание по правому краю
- center – по центру
- justify –выравнивание по ширине, для текста длина которого более чем одна строка

Имеет вертикальные margin по 16px

```html
<h1 align="center">заголовок</h1>
<h2 align="right">автор</h2>
<p align="justify">текст</p>
```

По умолчанию разделяются одной строкой

<!-- picture ------------------------------------------------------------------------------------------------------------------->

# picture (HTML5)

Позволяет осуществить загрузку изображений с настройками

атрибуты (для source):

- media для медиа выражения
- type

```html
<picture class="footer__logo">
  <!-- установим в зависимости от view порта -->
  <!-- современный вариант -->
  <source
    srcset="img/logo-green-small-1x.png 1x, img/logo-green-small-2x.png 2x"
    media="(max-width: 37.5em)"
  />
  <!-- запасной вариант -->
  <img
    srcset="img/logo-green-1x.png 1x, img/logo-green-2x.png 2x"
    alt="full logo"
    class="footer__logo"
    src="img/logo-green-2x.png"
  />
</picture>
```

<!-- progress ------------------------------------------------------------------------------------------------------------------->

# progress (HTML5)

Прогресс бар

Атрибуты:

- max
- value
-

<!-- q ------------------------------------------------------------------------------------------------------------------->

# q

строчная цитата, которая не требует новый абзац в отличает от [цитаты в абзаце](#blockquote)

Атрибут:

- cite

<!-- ruby ------------------------------------------------------------------------------------------------------------------->

# ruby rb rt

для аннотации текста,

<!-- s ------------------------------------------------------------------------------------------------------------------->

# s

для перечеркнутого текста, но для неактуального текста - del, есть устаревший - strike

<!-- script ------------------------------------------------------------------------------------------------------------------->

# script

Атрибуты:

- async - асинхронно загрузить скрипт, без src не сработает (async=false по умолчанию), если скрипт вставлен через document.createElement, будет вставлен асинхронно
- crossorigin
- defer - скрипт обрабатывается после загрузки документа, до события DOMContentLoaded, такие скрипты буду предотвращать это событие
- integrity - для безопасности, содержит метаданные
- nomodule - отключает возможность использования ES-6 модулей, можно использовать для старых браузеров

```html
<script type="module" src="main.mjs"></script>
<script nomodule src="fallback.js"></script>
```

- nonce - криптографический одноразовый номер
- src - url
- text - текстовое содержание
- type - по умолчанию js:
- - module - скрипт является модулем
- - importmap - скрипт является алиасом импортов

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

<!-- section ------------------------------------------------------------------------------------------------------------------->

# section (HTML5)

Структурный тег. Может включать в себя несколько article или наоборот

- у каждого section должен быть h1-h6

Примеры улучшения семантики:

```html
<!-- div -->
<div>
  <h1>Заголовок</h1>
  <p>Много замечательного контента</p>
</div>

<!-- div -->
<section>
  <h1>Заголовок</h1>
  <p>Много замечательного контента</p>
</section>

<div>
  <h2>Заголовок</h2>
  <img src="bird.jpg" alt="птица" />
</div>

<section>
  <h2>Заголовок</h2>
  <img src="bird.jpg" alt="птица" />
</section>
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

<!-- small ------------------------------------------------------------------------------------------------------------------->

# small (str)

уменьшает на 1 у.е

<!-- source ------------------------------------------------------------------------------------------------------------------->

# source (str HTML5)

Указывает разные варианты для [picture](#picture) [video](#video) [audio](#audio)

Атрибуты:

- адрес на ресурс, дескриптор ширины, и DPI
- sizes - работает только внутри picture
- src
- srcset - набор изображений
- type - MIME
- media - для медиа выражение, работает только с picture

<!-- span ------------------------------------------------------------------------------------------------------------------->

# span (str)

для определения строчного элемента внутри документа

<!-- strong ------------------------------------------------------------------------------------------------------------------->

# strong (str)

позволяет выделить текст заключенный в тег. Разница с b - strong для более значимого контента, b - для привлечения внимания. em - на него делается более сильный акцент

<!-- style ------------------------------------------------------------------------------------------------------------------->

# style

Атрибуты:

- type - mime
- media - для какого типа
- scoped - если указан, то стиль применится только внутри родительского элемента
- title
- disabled

```html
<article>
  <div>
    Атрибут scoped позволяет включить элементы стиля в середине документа.
    Внутренние правила применяются только внутри родительского элемента.
  </div>
  <p>
    Этот текст должен быть чёрным. Если он красный, ваш браузер не поддерживает
    атрибут scoped.
  </p>
  <section>
    <style scoped>
      p {
        color: red;
      }
    </style>
    <p>Этот должен быть красным.</p>
  </section>
</article>
```

<!-- sub sup ------------------------------------------------------------------------------------------------------------------->

# sub sup (str)

для надстрочного и подстрочного индексов

<!-- summary ------------------------------------------------------------------------------------------------------------------->

# summary (HTML5)

<!-- table, tr, th, td --------------------------------------------------------------------------------------------------------->

# table (block), tr, th, td

table - контейнер для таблицы

td - ячейка, table data
th - для создания одной ячейки таблицы, которая будет обозначена как заглавная, шрифт – жирный, выравнивание по центру
tr контейнер для создания строки таблицы

```html
<table _border="1" width="100%" cellpadding="5">
  <tr>
     
    <th>Ячейка 1</th>
    <!-- жирный шрифт, выравнивание по центру     -->
    <th>Ячейка 2</th>
    <!-- жирный шрифт, выравнивание по центру     -->
  </tr>
     
  <tr>
         
    <td>Ячейка 3</td>
    <!-- обычное форматирование -->
    <td>Ячейка 4</td>
    <!-- обычное форматирование -->
  </tr>
</table>
```

Атрибуты table:

- align - задает выравнивание по краю окна браузера, допустимые значения left, center, right (лучше использовать css)
- bgcolor – цвет заливки (лучше использовать css)
- border – толщина границы в пикселях (лучше использовать css)
- cellpading - определяет расстояние между границей ячейки и ее содержимым, добавляет пустое пространство к ячейке
- cellspacing – задает расстояние между внешними границами ячеек, border принимается расчет
- cols – указывает количество столбцов, помогая загрузки таблицы
- height – высота яичек, при размере меньше, чем факт браузер выставит самостоятельно
- объединение ячеек:
- - rowspan – Объединение ячеек по вертикали
- - colspan – устанавливает число ячеек, которые должны быть объединены по горизонтали
- rules – отображение границ между ячейками, значения (лучше использовать css):
- - cols (между колонами)
- - rows (строками)
- - group, которые определяются наличием тегов thead tfoot tbody colgroup col толщина границы задается с помощью атрибута border
- scope - добавляется к элементу th, сообщает скринридеру какие ячейки являются заголовками, принимает значения:
- - col
- - row
- - colgroup
- - rowgroup
- width – задает ширину таблицы
- id и header позволяют установить взаимодействие между заголовком и ячейками:
- - id - устанавливаем для каждого th
- - headers - для каждого td элемента, в качестве значения строка с id всех заголовков, к которым относится данный header

## td

Может использоваться только внутри table

Атрибуты tr:

- colspan - сколько столбцов нужно объединить, значения выше 1000 - игнорируются
- headers - список строк, каждая из которых соответствует id элементов th, список, если где-то использовался rowspan
- rowspan - объединение рядов, не выше 65534

```html
<thead>
  <tr>
    <!-- колонка purchase -->
    <th id="purchase">Purchase</th>
    <th id="location">Location</th>
    <th id="date">Date</th>
    <th id="evaluation">Evaluation</th>
    <th id="cost">Cost (€)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <!-- ряд haircut -->
    <th id="haircut">Haircut</th>
    <!-- первый элемент таблица соответствует заголовкам location и haircut -->
    <td headers="location haircut">Hairdresser</td>
    <td headers="date haircut">12/09</td>
    <td headers="evaluation haircut">Great idea</td>
    <td headers="cost haircut">30</td>
  </tr>

  ...
</tbody>
```

## tfoot

Используется для улучшения доступности, при печати документа. Используются вместе с tr и colspan. Встраивается в конце, после tbody

## th

Может быть использован только внутри элемента tr

Атрибуты:

- abbr - краткое описание содержимого
- colspan
- headers - список id тегов th
- rowspan
- scope:
- - row - если это заголовок строки,
- - col - если это заголовок ряда. rowgroup, colgroup

## thead

Находится после caption, colgroup

## вложенные таблицы

Таблицы можно встраивать друг в друга

```html
<table
  width="200"
  _bgcolor="###"
  cellspacing="0"
  cellpadding="5"
  _border="1"
  _align="right"
>
  <tr>
    <td>Содержимое таблицы</td>
  </tr>
  <p>…если бы здесь был текст, то он обтекал бы таблицу выше</p>
</table>
```

## Объединение ячеек

```html
<!-- Неверное объединение -->
<table _border="1" cellspadding="5" width="100%">
  <!-- //результатом будет таблица 3*2, с пустой ячейкой (3;2) -->
  <tr>
    <!-- //в ряд объединяем 2ячейки -->
    <td colspan="2">Ячейка 1</td>
    <!-- //третья в ряду -->
    <td>Ячейка 2</td>
  </tr>
  <tr>
    <td>Ячейка 3</td>
    <td></td>
    <td>Ячейка 4</td>
    <td>//2 ячейки во втором ряду</td>
  </tr>
</table>
```

## col и colgroup

используются для стилизации колонок. col находится внутри colgroup. Если определено несколько colgroup без col то поведение такое-же как и несколько col

Атрибуты col:

- span - сколько столбцов будет стилизовано

```html
<table>
  <colgroup>
    <!-- пустой тег col не применит стили к первому столбцу-->
    <col />
    <!-- во втором столбце будет применен background-color-->
    <col style="background-color: yellow" />
  </colgroup>
  <tr>
    <th>Data 1</th>
    <th>Data 2</th>
  </tr>
  <tr>
    <td>Calcutta</td>
    <td>Orange</td>
  </tr>
  <tr>
    <td>Robots</td>
    <td>Jazz</td>
  </tr>
</table>
```

## caption

заголовок таблицы

```html
<table>
  <caption>
    Заголовок таблицы
  </caption>
  <tr>
    <th></th>
    <th></th>
  </tr>
</table>
```

## thead, tbody, tfoot

Теги определяющие структуру таблицы

- thead - первая строка, должен быть после col, colgroup
- tfoot - последняя строка таблицы
- tbody - неявно встраивается во все таблицы (если его нет, но указать стиль в css, то стили добавятся), можно использовать несколько, если таблица большая

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

<!-- time  ------------------------------------------------------------------------------------------------------------------->

# time (HTML5)

для отображения времени

```html
<!-- Стандартная дата -->
<time datetime="2020-01-20">20 Января 2020</time>
<!-- Только год и месяц -->
<time datetime="2020-01">Январь 2020</time>
<!-- Только месяц и день -->
<time datetime="01-20">20 Января</time>
<!-- Только время, часы и минуты -->
<time datetime="19:30">19:30</time>
<!-- Также вы можете отобразить секунды и миллисекунды! -->
<time datetime="19:30:01.856">19:30:01.856</time>
<!-- Дата и время -->
<time datetime="2020-01-20T19:30">7.30pm, 20 Января 2020</time>
<!-- Дата и время со смещением по часовому поясу -->
<time datetime="2020-01-20T19:30+01:00"
  >7.30pm, 20 Января 2020, — это 8.30pm во Франции.</time
>
<!-- Вызов номера недели -->
<time datetime="2020-W04">Четвёртая неделя 2020</time>
```

<!-- title ------------------------------------------------------------------------------------------------------------------->

# title

используется в head

<!-- track ------------------------------------------------------------------------------------------------------------------->

# track

Встраиваемая дорожка в video и audio

- default
- kind:
- - subtitles
- - captions
- - descriptions
- - chapters
- - metadata
- label
- src
- srclang

```html
<video controls poster="/images/sample.gif">
  <source src="sample.mp4" type="video/mp4" />
  <source src="sample.ogv" type="video/ogv" />
  <track kind="captions" src="sampleCaptions.vtt" srclang="en" />
  <track kind="descriptions" src="sampleDescriptions.vtt" srclang="en" />
  <track kind="chapters" src="sampleChapters.vtt" srclang="en" />
  <track kind="subtitles" src="sampleSubtitles_de.vtt" srclang="de" />
  <track kind="subtitles" src="sampleSubtitles_en.vtt" srclang="en" />
  <track kind="subtitles" src="sampleSubtitles_ja.vtt" srclang="ja" />
  <track kind="subtitles" src="sampleSubtitles_oz.vtt" srclang="oz" />
  <track kind="metadata" src="keyStage1.vtt" srclang="en" label="Key Stage 1" />
  <track kind="metadata" src="keyStage2.vtt" srclang="en" label="Key Stage 2" />
  <track kind="metadata" src="keyStage3.vtt" srclang="en" label="Key Stage 3" />
  <!-- Fallback -->
  ...
</video>
```

<!-- u ------------------------------------------------------------------------------------------------------------------->

# u

подчеркнутый текст волнистой линией

<!-- ul ------------------------------------------------------------------------------------------------------------------->

# ul (block)

имеет верхний и нижний margin по 16px === 1em, padding-left === 1.5 em (40px)

!!!Отступы добавляются автоматически

```html
<!-- Список с маркерами в виде круга  -->
<ul type="disc"></ul>
<!-- Список с маркерами в виде окружностей  -->
<ul type="circle"></ul>
<!-- Список с маркерами в виде квадратов  -->
<ul type="square"></ul>
```

<!-- video ------------------------------------------------------------------------------------------------------------------->

# video (HTML5)

атрибуты:

- autobuffer -
- buffered -
- controls - отображать элементы управления
- width="400" -
- height="400" -
- autoplay - автозапуск видео
- loop - зацикленность
- muted - воспроизвести без звука
- poster - изображение до воспроизведения
- preload принимает значения:
- - none не буферизирует файл
- - auto буферизирует медиафайл
- - metadata буферирует только метаданные файла

```html
<video class="bg-video__content" autoplay muted loop>
  <!-- два формата для поддержке браузеров -->
  <source src="img/video.mp4" type="video/mp4" />
  <source src="img/video.webm" type="video/webm" />
  Your browser is not supported!
</video>

<!-- другой пример встраивания -->

<video
  src="rabbit320.webm"
  controls
  width="400"
  height="400"
  autoplay
  loop
  muted
  poster="poster.png"
>
  <p>
    Ваш браузер не поддерживает HTML5 видео. Используйте (это резервный контент)
    <a href="rabbit320.webm">ссылку на видео</a> для доступа.
  </p>
</video>
```

<!-- wbr ------------------------------------------------------------------------------------------------------------------->

# wbr (HTML5)

для переноса слов

```html
<div id="example-paragraphs">
  <p>Fernstraßenbauprivatfinanzierungsgesetz</p>
  <p>Fernstraßen<wbr />bau<wbr />privat<wbr />finanzierungs<wbr />gesetz</p>
  <p>Fernstraßen&shy;bau&shy;privat&shy;finanzierungs&shy;gesetz</p>
</div>
```
