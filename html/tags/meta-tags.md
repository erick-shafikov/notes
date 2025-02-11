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

<!-- link ----------------------------------------------------------------------------------------------------------------------->

# link

Показывает отношение сайта и внешних ссылок

Атрибуты:

- as - указывает тип контента требуется только rel="preload" или rel="prefetch" позволяет установить приоритет
- - audio
- - document (iframe)
- - embed
- - fetch (К JavaScript API XMLHttpRequest)
- - font
- - image (атрибут srcset)
- - object
- - script
- - style
- - track
- - video
- - worker
- crossorigin: anonymous, use-credentials
- href - url ресурса
- hreflang - язык
- importance: auto, high, low используется только rel="preload" или rel="prefetch"
- media - запрос для ресурса используется для внешних стилей
- referrerpolicy - no-referrer, no-referrer-when-downgrade, origin, origin-when-cross-origin, unsafe-url
- rel - определяет отношение ресурса и внешней ссылки
- - alternate - для указания ссылки на файл в формате XML
- - author - ссылка на автора
- - dns-prefetch
- - help - ссылка на справку
- - icon - адрес картинки
- - license - ссылка на лицензию
- - next - документ является частью блока документов, указывается ссылка на следующий документ
- - pingback
- - preconnect
- - prefetch
- - preload
- - prerender
- - prev - документ является частью блока документов, указывается ссылка на следующий документ
- - search - ссылка на ресурс, с поиском по данному сайту
- - stylesheet - стили
- sizes - только для иконок, только при rel=icon
- title - MIME тип

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
<!-- также допускается внутри Body -->
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
<!-- иконки для устройств apple -->
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

```html
<!-- добавление иконки для разных устройств (apple) -->
<!-- size определяет размер иконки -->
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
<link rel="apple-touch-icon" href="touch-icon-iphone.png" />
<link rel="apple-touch-icon" sizes="72x72" href="touch-icon-ipad.png" />
<link rel="apple-touch-icon" sizes="114x114" href="touch-icon-iphone4.png" />
<link rel="apple-touch-startup-image" href="/startup.png" />

<link rel="apple-touch-icon" type="image/png" href="/apple-touch-icon.png" />
```

```html
<!-- HTML Link Tags -->
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
```

## meta. Базовые теги

```html
<!-- лучше в теге html использовать lang -->
<meta http-equiv="Cache-Control" content="no-cache" />
<meta http-equiv="content-language" content="no-cache" />
<meta http-equiv="Content-Security-Policy" />
<meta http-equiv="content-type" content="mime-type" />
<meta http-equiv="Expires" content="0" />
<meta http-equiv="Pragma" content="no-cache" />
<meta http-equiv="set-cookie" content="определяет куки для страницы" />
<meta http-equiv="refresh" content="3;url=https://www.mozilla.org" />
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
<meta name="apple-touch-fullscreen" content="yes" />
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
<!-- 
 width - ширина экрана - 
 height- высота для области просмотра
 initial-scale - масштабирование
 minimum-scale - ограничения масштабирования
 maximum-scale - максимальный уровень масштабирования
 user-scalable- запрет на масштабирование при user-scalable=no
 interactive-widget=resizes-visual | resizes-content | overlays-content отображение доп элементов (клавиатуры)
-->
```

<!-- noscript ------------------------------------------------------------------------------------------------------------------->

# noscript

дочерние элементы этого тега будут отображаться есть нет поддержки js

- находится в head

<!-- script ------------------------------------------------------------------------------------------------------------------->

# script

Атрибуты:

- async - Это логический атрибут, указывающий браузеру, если возможно, загружать скрипт, указанный в атрибуте src, асинхронно.
- crossorigin
- defer - Это логический атрибут, указывающий браузеру, что скрипт должен выполняться после разбора документа, но до события DOMContentLoaded. Скрипты с атрибутом defer будут предотвращать запуск события DOMContentLoaded до тех пор, пока скрипт не загрузится полностью и не завершится его инициализация.
- integrity - для безопасности, содержит метаданные
- nomodule - отключает возможность использования ES-6 модулей, можно использовать для старых браузеров
- nonce - криптографический одноразовый номер
- text - текстовое содержимое элемента

```html
<script type="module" src="main.mjs"></script>
<script nomodule src="fallback.js"></script>
```

- text - текстовое содержание
- type - по умолчанию js:
- - module - скрипт является модулем
- - importmap - скрипт является алиасом импортов

<!-- style ------------------------------------------------------------------------------------------------------------------->

# style

содержит информацию о стилях документа или его части

- должен быть внутри head

Атрибуты:

- blocking
- type (устаревший) - mime
- media - для какого типа (медиа выражения, по умолчанию all)
- nonce - Криптографический одноразовый номер,
- scoped - если указан, то стиль применится только внутри родительского элемента
- title - задает изолированный стиль (работает только в ff)
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
- использовать лучше фразы, а не одно-два слова поисковые системы, как правило, отображают примерно 55-60 первых символов
- является для заголовками при добавление в избранное
- отображается в поисковых системах
- должен быть уникальным для всего сайта

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
