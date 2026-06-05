# Методы экземпляра

# atob

# alert()

# back()

# blur()

# btoa()

# cancelIdleCallback()

# clearInterval()

# clearTimeout()

# captureEvents()

# clearImmediate()

# clearInterval()

# setInterval()

# clearTimeout()

# setTimeout()

# close()

# confirm()

# createImageBitmap()

# disableExternalCapture()

# dispatchEvent()

# dump()

# fetch()

# enableExternalCapture()

# find()

# focus()

# forward()

# getAttention()

# getAttentionWithCycleCount()

# getComputedStyle()

При необходимости узнать размер, отступы, цвет элемента из CSS, а не только из атрибута style. Свойство Style оперирует только значением атрибута style бtз учета CSS-каскада. Есть два типа значений стиля:

- Вычисленное (computed) - это вычисленные значения после применения всех CSS правил но в относительных единицах, если такие есть rem, em
- Окончательное (resolved) - это значения в пикселях

```html
<head>
  <!-- из-за того, что стиль описан в глобальном стиле мы не сможем прочитать значения -->
  <style>
    body {
      color: red;
      margin: 5px;
    }
  </style>
</head>
<body>
  красный текст
  <script>
    alert(document.body.style.color); //пусто
    alert(document.body.style.marginTop); //пусто
  </script>
</body>
```

синтаксис getComputedStyle(element, [pseudo]) - результат вызова – объект со стилями похожий на elem.style
element – элемент для которого нужно получить значение
pseudo – указывается, если нужен стиль псевдоэлемента

```html
<head>
  <style>
    body {
      color: red;
      margin: 5px;
    }
  </style>
</head>
<body>
  <script>
    let computedStyle = getComputedStyle(document.body);
    alert(computedStyle.marginTop); //5px
    alert(computedStyle.color); //rgb(255, 0, 0)
  </script>
</body>
```

Вычисленное (computed) значение – это, то которое получено после применения всех CSS правил и CSS свойств наследования в относительных величинах. Окончательное(resolved) – непосредственно применяемое к элементу. getComputedStyle – возвращает окончательное значение стиля, требует полного наименования свойства, стили примененные к посещенным ссылкам - игнорируются

# getDefaulComputedStyle()

# getSelection()

# home() Не стандартно

# matchMedia()

```js
// для работы с медиа выражениям в js
var mediaQueryList = window.matchMedia("(orientation: portrait)");

if (mediaQueryList.matches) {
  /* Окно просмотра в настоящее время находится в книжной ориентации */
} else {
  /* Окно просмотра в настоящее время находится в альбомной ориентации */
}

var mediaQueryList = window.matchMedia("(orientation: portrait)"); // Создание списка выражений.
function handleOrientationChange(evt) {
  // Определение колбэк-функции для обработчика событий.
  if (evt.matches) {
    /* Окно просмотра в настоящее время находится в книжной ориентации */
  } else {
    /* Окно просмотра в настоящее время находится в альбомной ориентации */
  }
}

mediaQueryList.addListener(handleOrientationChange); // Добавление колбэк-функции в качестве обработчика к списку выражений.

handleOrientationChange(mediaQueryList); // Запуск обработчика изменений, один раз.
mediaQueryList.removeListener(handleOrientationChange);
```

# maximize()

# minimize()

# moveBy()

# moveTo()

# mozRequestAnimationFrame()

# open()

```ts
function open(
  url?: string, //нужный ресурс
  target?: string, // "_blank" | "_self"  | "_parent" | "_top," | либо строка которая будет являться названием окна
  windowFeatures?: string, // строка, которая содержит функции окна формата name=value
): WindowProxy | null;
```

Параметры строки windowFeatures:

- attributionsrc
- popup
- width или innerWidth
- height или innerHeight
- left или screenX
- top или screenY
- noopener
- noreferrer

Возвращает WindowProxy нового окна, если успешно отурыто окно, null если нет. Если открытый контекст не из того же источника, то скрипт не сможет взаимодействовать

```js
const windowFeatures = "left=100,top=100,width=320,height=320";
const handle = window.open(
  "https://www.mozilla.org/",
  "mozillaWindow",
  windowFeatures,
);
if (!handle) {
  // The window wasn't allowed to open
  // This is likely caused by built-in popup blockers.
  // …
}
```

# openDialog()

# postMessage()

# print()

# prompt()

# releaseEvents()

# removeEventListener()

# requestIdleCallback() Экспериментальная возможность

# resizeBy()

# resizeTo()

# restore()

# routeEvent()

# scroll(), scrollBy(), scrollTo()

```js
window.scrollBy(x, y); //– прокручивает страницу относительно ее текущего положения
window.scrollTo(pageX, pageY); //– прокручивает страницу на абсолютные координаты, чтобы прокрутить в самое начало window.scrollTo(0,0)
//или с настройками
window.scrollTo({
  top: 100,
  left: 0,
  behavior: "smooth",
});
elem.scrollIntoView(true); //– прокручивает страницу так, чтобы элемент оказался сверху при top = true и внизу если top = false
// или с настройками
this.scrollIntoView({
  behavior: "smooth",
  block: "end",
  inline: "nearest",
});
document.body.style.overflow = "hidden"; //выключает прокрутку
document.body.style.overflow = ""; //включает ее обратно
```

# scrollByLines()

# scrollByPages()

# setInterval()

# setTimeout()

# setCursor()

# setImmediate()

# setInterval()

# setResizable

# setTimeout()

# showModalDialog()

# sizeToContent()

# stop()

# updateCommands()
