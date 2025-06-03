# Советы по улучшению производительности

## Стили

- Подключить в шапке сайта стили для загрузки первого экрана (критические стили) грузить в шапке, потом перейти к пункту

```html
<head>
 <style id="critical">
 /* критические стили */
 </style>
</head>
<body>
 <script>
 function onStyleLoad() {
 // блок критических стилей удаляется после загрузки файла стилей
 document.getElementById("critical").remove();
 }
 </script>
 <!-- ссылка на файл стилей ближе к концу body —>
 <link rel=”stylesheet” href=”bundle.css” onload=”onStyleLoad();” />
</body>
```

- Критические стили можно выделить с помощью утилит

- Подгруздка стилей по ссылке через JS (создать link и присвоить атрибуты src к стилям) Минус – будет штраф по CLS
- Выбирать из библиотек нужные куски с помощью препроцессоров
- минификация

# Шрифты

- загрузка шрифтов в двух вариантах (font-display: swap/noswap атрибут, который определяет как загружается шрифт)
- Загрузить шрифты можно в двух направлениях – можно загрузить асинхронно, можно выбрать похожий на нужный, что бы предотвратить CLS

## Скрипты

- Атрибут defer
- минификация
- разбиение длинных задач(более 50 мс)
- код на отдельные бандлы
- предварительная загрузка ключевых ресурсов

```html
<link rel="preload" as="script" href="script.js" />
<link rel="preload" as="style" href="style.css" />
```

- кеширование c помощью Service Workers

```js
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open("v1").then((cache) => {
      return cache.addAll(["/critical.css", "/main.js"]);
    })
  );
});
```

-tree shaking в скриптах (настройка в бандлере)

## Виджеты

- iframe можно заменить картинкой и загружать по клику
- предварительно загружать ресурсы

```html
<link rel="preconnect" href="https://example.com" />
```

# BP. Работа с изображениями

Оптимизация загрузки большого количества изображений

- Зарезервировать место для img с помощью width и height

```html
<img
  src="pillow.jpg"
  width="640"
  height="360"
  alt="purple pillow with flower pattern"
/>
```

- Ленивая отрисовка
- использование селекторов srcset (позволяет указать какую картинку загружать в зависимости от VP) и size работает как медиа-запрос

```html
<picture>
  <source media="(min-width: 768px)" srcset="tablet_image.png" />
  <source media="(min-width: 1024px)" srcset="desktop_image.png" />
  <img src="mobile_image.png" alt="" />
</picture>
```

- ленивая загрузка img loading === lazy дял изображений второго порядка

```html
<img
  src="pillow.jpg"
  width="640"
  height="360"
  alt="purple pillow with flower pattern"
  loading="”lazy”"
/>
```

- отложенные изображения для изображений второго

```html
<link
  rel="”preload”"
  as="”image”"
  href="”mobileBanner.png”"
  media="(max-width:768px)"
/>
<link
  rel="”preload”"
  as="”image”"
  href="”desktopBanner.png”"
  media="(min-width:1600px)"
/>
```

- Размытая заглушка с помощью background-image
- сжатие изображений
- форматы wbeP и avif

```html
<picture>
  <source srcset="image.avif" type="image/avif" />
  <source srcset="image.webp" type="image/webp" />
  <img src="image.jpg" alt="..." width="800" height="600" />>
</picture>
```

- gif в видео

```html
<video autoplay loop muted playsinline>
  <source src="animation.webm" type="video/webm" />
  <source src="animation.mp4" type="video/mp4" />
</video>
```

# сервер

- http3
- Настройка Brotli-сжатия на сервере.
