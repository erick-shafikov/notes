<!-- Добавление стилей ----------------------------------------------------------------------------------------------------------------------->

# Добавление стилей. Связанные стили

Связанные стили – описание селекторов и их значение располагается в отдельном файле с расширением css, с использованием тега <link>, который помещается в контейнер <head>, варианты подключения:

1. Вариант:

```html
<head>
   
  <meta />
   
  <title>…</title>
    <link rel=stylesheets href=./myStyle.css>
  <!--можно подключить стили, которые лежат на компе-->
   
  <link rel="stylesheets" href="www…" />
  <!-- подключить стили с другого сайта -->
</head>
```

2. Вариант:
   С помощью JS, создаем скрипт с тегом link или style

## Глобальные и внутренние стили

### Глобальный стиль(в head)

```html
<head>
  <meta />
   
  <title></title>
  <style>
    h1{
      background-color: red
    }       
  </style>
</head>
```

### Внутренний стиль

```html
<head>
   
  <meta />
   
  <title></title>
</head>
   
<p style="color:123">…</p>
```

Приоритет (по убыванию) внутренний, глобальный, связанный

- [импорт CSS поддерживает разные условия](./at-rules.md/#import)

```html
<head>
  <meta />
  <title>…</title>
  <style>
    @import /style/main.css screen; /* Стиль для монитора */
    @import /style/smart.css print, handheld; /* Стиль для печати и смартфона */
  </style>
</head>
```

media позволяет указать тип носителя

```html
<link media="print" handheld rel="stylesheet" href="pront.css" />
<link media="screen" rel="stylesheet" href="main.css" />
```

## Подключение

```scss
//Произвольные строки:
{
  font-family: "Times New Roman", serif
  content: привет
}
// CSS-директивы
@font-face {
  font-family: Open Sans;
  src:
    url(),
    url();
}
@media (max-width: 600px) {//при максимальное ширине отображения sidebar скроется
  .sidebar {
    display: none;
  }
}

```

<!-- @document ------------------------------------------------------------------------------------------------------------------------------->

# @document

ограничение по стилю в зависимости от url

```scss
@document url("https://www.example.com/")
{
  h1 {
    color: green;
  }
}
```

- url(), который соответствует точному URL-адресу.
- url-prefix(), который совпадает, если URL-адрес документа начинается с указанного значения.
- domain(), который совпадает, если URL-адрес документа находится в предоставленном домене (или его субдомене).
- media-document(),с параметром видео, изображения, плагина или всего.
- regexp(), который совпадает, если URL-адрес документа сопоставляется с предоставленным регулярным выражением. Выражение должно соответствовать всему URL-адресу.

<!-- @import --------------------------------------------------------------------------------------------------------------------------------->

# @import

Позволяет импортировать стили, должно быть на верху фала, кроме @charset

```scss
//путь до файла
@import "url";
@import "custom.css";
@import url("chrome://communicator/skin/");
//для разных устройств
@import url("fineprint.css") print;
@import url("bluish.css") print, screen;
@import "common.css" screen;
@import url("landscape.css") screen and (orientation: landscape);
//с учетом медиа запросов
@import url("gridy.css") supports(display: grid) screen and (max-width: 400px);
@import url("flexy.css") supports((not (display: grid)) and (display: flex)) screen
  and (max-width: 400px);
// supports
@import url("whatever.css") supports((selector(h2 > p)) and
    (font-tech(color-COLRv1)));
// c использованием layer
@import "theme.css" layer(utilities);
```

Пример с layer

```scss
@import url(headings.css) layer(default);
@import url(links.css) layer(default);

@layer default {
  audio[controls] {
    display: block;
  }
}
```

## нормализация

Нормализация – это файл CSS файл, который обеспечивает кроссбраузерность подключение:

```html
<link rel=stylesheet href=css/normalize.css>
```

Сброс – сброс стилей
Подключение нестандартных шрифтов, Google fonts для шрифтов

сброс стилей для форм.

```scss
button,
input,
select,
textarea {
  font-family: inherit;
  font-size: 100%;
  box-sizing: border-box;
  padding: 0;
  margin: 0;
}

textarea {anchor-name
  overflow: auto;
}
```
