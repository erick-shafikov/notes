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
- height
- loop
- muted
- preload
- - none
- - metadata
- - auto
- src - можно использовать вместо source
- width

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

```html
<figure>
  <figcaption>Listen to the T-Rex:</figcaption>
  <audio controls src="/media/cc0-audio/t-rex-roar.mp3"></audio>
  <a href="/media/cc0-audio/t-rex-roar.mp3"> Download audio </a>
</figure>
```

если нет атрибута control то визуальных элементов не будет, если задан, то будет отображаться обычный inline элемент, который позволяет стилизовать только контейнер, но не элементы управления

Доступны для CSS border и border-radius, padding, margin

<!-- canvas  ----------------------------------------------------------------------------------------------->

# canvas (block, HTML5)

Область для отрисовки

Атрибуты:

- height - 150 по умолчанию
- width - 300 по умолчанию
- moz-opaque - полупрозрачность

<!-- figure figcaption----------------------------------------------------------------------------------------------------------->

# figure и figcaption (block, HTML5)

Потоковый элемент. Тег картинки и подписи к ней. теги нужны для улучшения семантики

figure - не является изображением, может быть несколькими изображениями, куском кода, аудио, видео, уравнением, таблицей, либо чем-то другим.

```html
<figure class="story__shape">
  <img src="img/nat-8.jpg" alt="person on a tour" class="story__img" />
  <figcaption class="story__caption">Mary Smith</figcaption>
</figure>
```

```html
<!-- смысл чтобы преобразовать вот это: -->
<div class="figure">
  <img
    src="images/dinosaur.jpg"
    alt="The head and torso of a dinosaur skeleton;
            it has a large head with long sharp teeth"
    width="400"
    height="341"
  />

  <p>A T-Rex on display in the Manchester University Museum.</p>
</div>
<!-- в это: -->
<figure>
  <img
    src="images/dinosaur.jpg"
    alt="The head and torso of a dinosaur skeleton;
            it has a large head with long sharp teeth"
    width="400"
    height="341"
  />

  <figcaption>
    A T-Rex on display in the Manchester University Museum.
  </figcaption>
</figure>
```

Пример с кодом

```html
<figure>
  <figcaption>Get browser details using <code>navigator</code>.</figcaption>
  <pre>
function NavigatorExample() {
  var txt;
  txt = "Browser CodeName: " + navigator.appCodeName + "; ";
  txt+= "Browser Name: " + navigator.appName + "; ";
  txt+= "Browser Version: " + navigator.appVersion  + "; ";
  txt+= "Cookies Enabled: " + navigator.cookieEnabled  + "; ";
  txt+= "Platform: " + navigator.platform  + "; ";
  txt+= "User-agent header: " + navigator.userAgent  + "; ";
  console.log("NavigatorExample", txt);
}
  </pre>
</figure>
```

Цитирование

```html
<figure>
  <figcaption><cite>Edsger Dijkstra:</cite></figcaption>
  <blockquote>
    If debugging is the process of removing software bugs, then programming must
    be the process of putting them in.
  </blockquote>
</figure>
```

<!-- iframe ------------------------------------------------------------------------------------------------------------------->

# iframe

Нужен для отображения другой страницы в контексте текущей

Фреймы – разделяют окно браузера на отдельные области расположенные вплотную друг у другу, в каждый загружается отдельная веб страница. Позволяют открыть документ в одном фрейме по ссылке нажатой в совершенно в другом фрейме. поддерживают вложенную структуру

атрибуты:

- глобальные
- allow - какие функции доступны для frame
- - allow="fullscreen" - позволяет раскрыть frame на dtcm 'rhfy'
- allowfullscreen - возможность открыть фрейм в полноэкранном режиме
- frameborder - обозначить границу, значения 0 и 1, лучше использовать border
- loading:
- - eager - сразу ,
- - lazy - пока не дойдет до места просмотра
- name - для фокусировки
- referrerpolicy:
- - no-referrer - без заголовка Referer
- - no-referrer-when-downgrade - Referer только по https
- - origin - только на источник
- - origin-when-cross-origin
- - same-origin
- - strict-origin
- - strict-origin-when-cross-origin
- - unsafe-url
- width, height
- sandbox - повышает настройки безопасности
- - allow-downloads - позволяет загрузить файлы через тег а
- - allow-forms - отправка форм
- - allow-modals - позволяет показывать alert
- - allow-orientation-lock
- - allow-pointer-lock
- - allow-popups - Window.open(), target="\_blank",
- - allow-popups-to-escape-sandbox
- - allow-presentation
- - allow-same-origin - хранилище js
- - allow-scripts
- - allow-top-navigation
- - allow-top-navigation-by-user-activation
- - allow-top-navigation-to-custom-protocols
- src
- srcdoc

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

- все frame-ы лежат в объекте window.frames
- через contentWindow frame может получить доступ к window
- window.parent - ссылка на родительское окно
- Window.postMessage() - метод передачи сообщений

## Изменение размеров

для блокировки возможности изменения размера атрибут noresize

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

<!-- frameset ---------------------------------------------------------------------------------------------------------------------->

# frameset

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

<!-- img ---------------------------------------------------------------------------------------------------------------------->

# img (str)

Атрибуты:

- alt - альтернативный текст, если изображение не загрузилось
- crossorigin значения:
- - anonymous
- - use-credentials
- decoding - поведение декодирования:
- - auto
- - sync - синхронно с другим контентом
- - async - параллельно, что бы уменьшить задержку с другим контентом
- height - высота контейнера для изображения
- importance - приоритет загрузки auto, low, high
- ismap - карта ссылок, если img является потомком a
- intrinsicsize - игнорирование размера
- loading: eager, lazy
- referrerpolicy: no-referrer, no-referrer-when-downgrade, origin, origin-when-cross-origin, unsafe-url
- sizes - медиа выражения и слот в каждой строке, соответствие с srcset по принципу самый первый, который больше. одна или несколько строк разделенные запятыми, состоящие из медиа запроса, размер источника
- srcset - позволяет загружать картинки в зависимости от ширины экрана. Формат - ссылка, запятая пробел, где w - это ширина в пикселях
- title - лучше figure и figcaption, доп информация
- usemap
- width - ширина изображения

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

для связи заголовка и изображения [figure и figcaption](#figure-и-figcaption-block-html5)
для более гибкого адаптивного поведения [picture](#picture-html5)

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
    media="(max-width: 37.5em)"
    srcset="img/logo-green-small-1x.png 1x, img/logo-green-small-2x.png 2x"
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
  <!-- ссылка на титры в .vtt формате-->
  <track kind="subtitles" src="subtitles_en.vtt" srclang="en" />
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
