- (str) - строчный элемент
- (block) - блочный
- (HTML5) - семантический тег, каждый такой тег начинает с определения структурны он или поточный

<!-- article ----------------------------------------------------------------------------------------------------------------------->

# article (block, HTML5)

Структурный тег. создан для отдельной смысловой единице, которую можно оторвать от сайта. Размер текста 1.5 rem === 24px

Атрибуты:

- только глобальные

- должен быть идентифицирован добавляя теги h1-h6
- может быть вложен в section или в него могут быть вложены несколько article
- можно добавить время и автора

```html
<article class="film_review">
  <header>
    <h2>Парк Юрского периода</h2>
  </header>
  <section class="main_review">
    <p>Динозавры были величественны!</p>
  </section>
  <section class="user_reviews">
    <article class="user_review">
      <p>Слишком страшно для меня.</p>
      <footer>
        <p>
          Опубликовано
          <time datetime="2015-05-16 19:00">16 мая</time>
          Лизой.
        </p>
      </footer>
    </article>
    <article class="user_review">
      <p>Я согласен, динозавры мои любимцы.</p>
      <footer>
        <p>
          Опубликовано
          <time datetime="2015-05-17 19:00">17 мая</time>
          Томом.
        </p>
      </footer>
    </article>
  </section>
  <footer>
    <p>
      Опубликовано
      <time datetime="2015-05-15 19:00">15 мая</time>
      Стаффом.
    </p>
  </footer>
</article>
```

<!-- aside ----------------------------------------------------------------------------------------------------------------------->

# aside (block, HTML5)

Структурный тег. для неосновного контента. отступление предназначен для отступления. например пометки на полях в печатном
журнале

Атрибуты:

- глобальные

<!-- base ----------------------------------------------------------------------------------------------------------------------->

# base

Определяет основной url страницы, родители - head, body

```html
<head>
  <base href="https://www.w3schools.com/" target="_blank" />
</head>

<body>
  <img src="images/stickman.gif" width="24" height="39" alt="Stickman" />
  <a href="tags/tag_base.asp">HTML base Tag</a>
</body>
```

<!-- body --------------------------------------------------------------------------------------------------------------->

# body

Атрибуты:

- background фоновое изображение (лучше через css background)
- bgcolor - цвет фона (лучше через css background-color)
- цвет ссылок:
- - alink - цвет текста гиперссылок (лучше через css)
- - link - цвет непосещенных гиперссылок (лучше через css)
- - vlink - цвет посещенной ссылки (лучше через css)
- margin:
- - bottommargin - отступ внизу (лучше через css margin-bottom)
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

<!-- data ----------------------------------------------------------------------------------------------------------------->

# data (block, HTML5)

добавит машиночитаемый код, если это дата или время лучше использовать тег time

```html
<ul>
  <!-- каждый элемент связан со своим id -->
  <li><data value="398">Mini Ketchup</data></li>
  <li><data value="399">Jumbo Ketchup</data></li>
  <li><data value="400">Mega Jumbo Ketchup</data></li>
</ul>
```

<!-- div ----------------------------------------------------------------------------------------------------------------->

# div (block)

<!-- embed ---------------------------------------------------------------------------------------------------------------------->

# embed (block, HTML5)

устаревший вариант для встраивания контента, атрибуты: height, src, type, width

<!-- footer----------------------------------------------------------------------------------------------------------->

# footer (block, HTML5)

для контента внизу страницы, нижний колонтитул, на количество ограничения не накладываются

<!-- head ---------------------------------------------------------------------------------------------------------------------->

# head

Не отображается в документе, главная цель – метаданные, содержит:

title - отображает заголовок на странице

Внутри себя использует:

- [тег <link />](#link)
- [теги <meta />](#meta)
- [тег <script />](#script)

Создается автоматически, содержит:

- заголовок (title) страницы
- ссылки на файлы CSS (если вы хотите применить к вашему HTML стили CSS)
- ссылки на иконки
- другие метаданные (данные о HTML: автор и важные ключевые слова, описывающие документ.)

<!-- header ---------------------------------------------------------------------------------------------------------------------->

# header (block, HTML5)

Потоковый тег. Если это дочерний элемент body, то это заголовок всей страницы, также может быть заголовком section или article. Должен содержать h1-h6

```html
<article>
  <header>
    <h2>Планета Земля</h2>
    <p>Опубликовано в среду, 4 октября 2017, Джейн Смит</p>
  </header>
  <p>
    Мы живём на сине-зелёной планете, на которой до сих пор так много
    неизведанного.
  </p>
  <p>
    <a href="https://janesmith.com/the-planet-earth/">Продолжить чтение...</a>
  </p>
</article>
```

<!--  hgroup ---------------------------------------------------------------------------------------------------------------->

# hgroup (block, HTML5)

Группирует h1-h6 в один заголовок или группирует несколько тегов p

```html
<hgroup>
  <h1>Frankenstein</h1>
  <p>Or: The Modern Prometheus</p>
</hgroup>
<p>...</p>
```

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
<html lang="ru"></html>
```

выделяет символы

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

Потоковый тег.

- один на всю страницу,
- должен быть внутри body, при добавлении id позволяет упростить навигацию для устройств со спец возможностями.
- Не должен быть вложен в другие

<!-- meta ----------------------------------------------------------------------------------------------------------------------->

# meta

Синтаксис - атрибуты name и content
Существую og метаданные open graph для facebook, так же есть у твиттера

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

- name - не следует указывать, если установлены itemprop, http-equiv или charset
- - application-name
- - referrer
- - creator
- - googlebot
- - publisher
- - robots
- - scheme

## meta. кодировка страницы

```html
<meta charset="utf-8" />
<meta http-equiv="refresh" content="3;url=https://www.mozilla.org" />
```

## meta. Базовые теги

```html
<meta http-equiv="Expires" content="0" />
<meta http-equiv="Pragma" content="no-cache" />
<meta http-equiv="Cache-Control" content="no-cache" />
```

```html
<!-- Автор -->
<meta name="author" content="name, email@hotmail.com" />
<meta name="abstract" content="" />
<meta name="Classification" content="Business" />
<meta name="category" content="" />
<meta name="copyright" content="company name" />
<meta name="copyright" content="" />
<meta name="coverage" content="Worldwide" />
<!--  используется на страницах поисковой выдачи. в поисковом запросе будет находится в описании под ссылкой -->
<meta name="description" content="150 words" />
<meta name="designer" content="" />
<meta name="distribution" content="Global" />
<meta name="directory" content="submission" />
<meta name="identifier-URL" content="http://www.websiteaddress.com" />
<meta name="keywords" content="your, tags" />
<meta name="language" content="ES" />
<meta name="owner" content="" />
<meta name="rating" content="General" />
<meta name="robots" content="index,follow" />
<meta name="revised" content="Sunday, July 18th, 2010, 5:15 pm" />
<meta name="revisit-after" content="7 days" />
<meta name="reply-to" content="email@hotmail.com" />
<meta name="subject" content="your website's subject" />
<meta name="summary" content="" />
<meta name="topic" content="" />
<meta name="url" content="http://www.websiteaddrress.com" />
```

## meta. OpenGraph мета теги

используется для отображения ссылки на fb

```html
<meta name="og:title" content="The Rock" />
<meta name="og:type" content="movie" />
<meta name="og:url" content="http://www.imdb.com/title/tt0117500/" />
<meta name="og:image" content="http://ia.media-imdb.com/rock.jpg" />
<meta name="og:site_name" content="IMDb" />
<meta
  name="og:description"
  content="A group of U.S. Marines, under command of..."
/>
<meta name="fb:page_id" content="43929265776" />
<meta name="og:email" content="me@example.com" />
<meta name="og:phone_number" content="650-123-4567" />
<meta name="og:fax_number" content="+1-415-123-4567" />
<meta name="og:latitude" content="37.416343" />
<meta name="og:longitude" content="-122.153013" />
<meta name="og:street-address" content="1601 S California Ave" />
<meta name="og:locality" content="Palo Alto" />
<meta name="og:region" content="CA" />
<meta name="og:postal-code" content="94304" />
<meta name="og:country-name" content="USA" />
<meta property="og:type" content="game.achievement" />
<meta property="og:points" content="POINTS_FOR_ACHIEVEMENT" />
<meta property="og:video" content="http://example.com/awesome.swf" />
<meta property="og:video:height" content="640" />
<meta property="og:video:width" content="385" />
<meta property="og:video:type" content="application/x-shockwave-flash" />
<meta property="og:video" content="http://example.com/html5.mp4" />
<meta property="og:video:type" content="video/mp4" />
<meta property="og:video" content="http://example.com/fallback.vid" />
<meta property="og:video:type" content="text/html" />
<meta property="og:audio" content="http://example.com/amazing.mp3" />
<meta property="og:audio:title" content="Amazing Song" />
<meta property="og:audio:artist" content="Amazing Band" />
<meta property="og:audio:album" content="Amazing Album" />
<meta property="og:audio:type" content="application/mp3" />
```

Create Custom Meta Tags

```html
<meta name="google-analytics" content="1-AHFKALJ" />
<meta name="disqus" content="abcdefg" />
<meta name="uservoice" content="asdfasdf" />
<meta name="mixpanel" content="asdfasdf" />
```

Company/Service Meta Tags Apple Meta Tags

```html
<meta name="apple-mobile-web-app-capable" content="yes" />
<meta content="yes" name="apple-touch-fullscreen" />
<meta name="apple-mobile-web-app-status-bar-style" content="black" />
<meta name="format-detection" content="telephone=no" />
<meta
  name="viewport"
  content="width = 320, initial-scale = 2.3, user-scalable = no"
/>
```

## Internet Explorer Meta Tags

```html
<meta
  http-equiv="Page-Enter"
  content="RevealTrans(Duration=2.0,Transition=2)"
/>
<meta
  http-equiv="Page-Exit"
  content="RevealTrans(Duration=3.0,Transition=12)"
/>
<meta name="mssmarttagspreventparsing" content="true" />
<meta http-equiv="X-UA-Compatible" content="chrome=1" />
<meta name="msapplication-starturl" content="http://blog.reybango.com/about/" />
<meta name="msapplication-window" content="width=800;height=600" />
<meta name="msapplication-navbutton-color" content="red" />
<meta name="application-name" content="Rey Bango Front-end Developer" />
<meta name="msapplication-tooltip" content="Launch Rey Bango's Blog" />
<meta
  name="msapplication-task"
  content="name=About;action-uri=/about/;icon-uri=/images/about.ico"
/>
<meta
  name="msapplication-task"
  content="name=The Big List;action-uri=/the-big-list-of-javascript-css-and-html-development-tools-libraries-projects-and-books/;icon-uri=/images/list_links.ico"
/>
<meta
  name="msapplication-task"
  content="name=jQuery Posts;action-uri=/category/jquery/;icon-uri=/images/jquery.ico"
/>
<meta
  name="msapplication-task"
  content="name=Start Developing;action-uri=/category/javascript/;icon-uri=/images/script.ico"
/>
<!-- иконка -->
<link rel="shortcut icon" href="/images/favicon.ico" />
<!-- Для iPad 3 с Retina-экраном высокого разрешения: -->
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

TweetMeme Meta Tags

```html
<meta name="blogcatalog" />
```

Rails Meta Tags

```html
<meta name="csrf-param" content="authenticity_token" />
<meta
  name="csrf-token"
  content="/bZVwvomkAnwAI1Qd37lFeewvpOIiackk9121fFwWwc="
/>
```

Apple Tags

```html
<meta name="apple-mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-status-bar-style" content="black" />
<meta name="format-detection" content="telephone=no" />
<meta
  name="viewport"
  content="width = 320, initial-scale = 2.3, user-scalable = no"
/>
<meta name="viewport" content="width = device-width" />
<meta name="viewport" content="initial-scale = 1.0" />
<meta name="viewport" content="initial-scale = 2.3, user-scalable = no" />
<link rel="apple-touch-icon" href="touch-icon-iphone.png" />
<link rel="apple-touch-icon" sizes="72x72" href="touch-icon-ipad.png" />
<link rel="apple-touch-icon" sizes="114x114" href="touch-icon-iphone4.png" />
<link rel="apple-touch-startup-image" href="/startup.png" />

<link rel="apple-touch-icon" type="image/png" href="/apple-touch-icon.png" />
```

// HTML Link Tags

```html
<link
  rel="alternate"
  type="application/rss+xml"
  title="RSS"
  href="http://feeds.feedburner.com/martini"
/>
<link rel="shortcut icon" type="image/ico" href="/favicon.ico" />
<link rel="fluid-icon" type="image/png" href="/fluid-icon.png" />
<link rel="me" type="text/html" href="http://google.com/profiles/thenextweb" />
<link rel="shortlink" href="http://blog.unto.net/?p=353" />
<link rel="archives" title="May 2003" href="http://blog.unto.net/2003/05/" />
<link rel="index" title="DeWitt Clinton" href="http://blog.unto.net/" />
<link
  rel="start"
  title="Pattern Recognition 1"
  href="http://blog.unto.net/photos/pattern_recognition_1_about/"
/>
<link
  rel="prev"
  title="OpenSearch and OpenID?  A sure way to get my attention."
  href="http://blog.unto.net/opensearch/opensearch-and-openid-a-sure-way-to-get-my-attention/"
/>
<link rel="next" title="Not blog" href="http://blog.unto.net/meta/not-blog/" />
<link
  rel="search"
  href="/search.xml"
  type="application/opensearchdescription+xml"
  title="Viatropos"
/>
<link
  rel="self"
  type="application/atom+xml"
  href="http://www.syfyportal.com/atomFeed.php?page=3"
/>
<link rel="first" href="http://www.syfyportal.com/atomFeed.php" />
<link rel="next" href="http://www.syfyportal.com/atomFeed.php?page=4" />
<link rel="previous" href="http://www.syfyportal.com/atomFeed.php?page=2" />
<link rel="last" href="http://www.syfyportal.com/atomFeed.php?page=147" />
<link rel="shortlink" href="http://smallbiztrends.com/?p=43625" />
<link
  rel="canonical"
  href="http://smallbiztrends.com/2010/06/9-things-to-do-before-entering-social-media.html"
/>
<link
  rel="EditURI"
  type="application/rsd+xml"
  title="RSD"
  href="http://smallbiztrends.com/xmlrpc.php?rsd"
/>
<link rel="pingback" href="http://smallbiztrends.com/xmlrpc.php" />
<!-- для стилей -->
<link
  media="only screen and (max-device-width: 480px)"
  href="http://wordpress.org/style/iphone.css"
  type="text/css"
  rel="stylesheet"
/>
```

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

<!-- title ------------------------------------------------------------------------------------------------------------------->

# title

используется в head,

- является обязательным
- является для заголовками при добавление в избранное
- отображается в поисковых системах

```html
<!DOCTYPE html>
<html>
  <head>
    <title>HTML Elements Reference</title>
  </head>
  <body>
    <h1>This is a heading</h1>
    <p>This is a paragraph.</p>
  </body>
</html>
```
