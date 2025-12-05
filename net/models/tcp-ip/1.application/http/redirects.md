# перенаправление

нужны для:

- привязки разных доменов к одному источнику
- переезд на другой домен

это ответ с кодом статуса 3хх, браузер использует новый url и перенаправляет, бывают:

- постоянные:
- - 301 - Moved Permanently методы get
- - 308 - Permanent Redirect реорганизация не get запросов
- временные:
- - 302 Found - времена недоступна
- - 303 See Other - post put
- - 307 Temporary Redirect
- специальные:
- - 300 Multiple Choice
- - 304 Not Modified

Альтернативные варианты перенаправления:

- HTML перенаправления

```html
<head>
  <meta http-equiv="refresh" content="0; URL=http://www.example.com/" />
</head>
```

- JavaScript:

```js
window.location = "http://www.example.com/";
```

# условные запросы

запросы для валидации контента в кеше, виды:

- безопасные
- небезопасные

Реализуется с помощью заголовков:

- If-Match
- If-None-Match
- If-Modified-Since
- If-Unmodified-Since
- If-Range
- Last-Modified и ETagMD
