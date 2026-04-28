# заголовки ответа

CH - client hint

Заголовки по типу ответ от сервера:

# Accept-CH

заголовок, который отправляет сервер, что бы получить CH заголовки от браузера, возможные значения: Sec-CH-UA-Model, Sec-CH-UA-Form-Factors, DPR, Viewport-Width, Width - сервер требует доп информацию, браузер на каждый запрос будет отправлять Sec-CH-UA-Model, Sec-CH-UA-Form-Factors. Поведение можно реализовать в html:

```html
<meta http-equiv="Accept-CH" content="Width, Downlink, Sec-CH-UA" />
```

Пример запроса:

```bash
Accept-CH: Sec-CH-Viewport-Width, Sec-CH-Width # отправил сервер
Vary: Sec-CH-Viewport-Width, Sec-CH-Width # какие vary заголовки ждет сервер
```

# Activate-Storage-Access

Это позволяет серверу активировать предоставленное разрешение на доступ к своим неразделенным файлам cookie в межсайтовом запросе.

```bash
Activate-Storage-Access: retry; allowed-origin="https://foo.bar"
Activate-Storage-Access: retry; allowed-origin=*
Activate-Storage-Access: load
```

# Activate-Storage-Access

заголовок предоставляет доступ к своим кукам для кросс запросов, сервер использует Sec-Fetch-Storage-Access заголовок

# Age

сколько объект запроса находился в кеше прокси

# Allow

Список дял доступных методов, должен возвращаться для 405 ошибки (Not Allowed)

# Alt-Svc

Приоритетный альтернативный ресурс

# Clear-Site-Data

клиент должен очистить все данные

```bash
# Single directive
Clear-Site-Data: "cache"

# Multiple directives (comma separated)
Clear-Site-Data: "cache", "cookies"

# Wild card
Clear-Site-Data: "*"
```

Значения: cache, cookies, executionContexts, prefetchCache, prerenderCache, storage \*

# Content-Security-Policy

Определяет кто имеет доступ к ресурсу

```bash
Content-Security-Policy: <policy-directive>; <policy-directive>
```

Директивы и значения:

- запроса:
- - директивы: child-src, connect-src, default-src, fenced-frame-src, font-src, frame-src, img-src, manifest-src, media-src, object-src, prefetch-src, script-src, script-src-elem, script-src-attr, style-src, style-src-elem, style-src-attr, worker-src
- - значения:
- - - none - полная блокировка
- - - self - только со своего origin
- - - host-source - url или ip адрес
- - - scheme-source - http или ws
- - - nonce-nonce_value,
- - - hash_algorithm-hash_value
- Директивы документа:
- - base-uri, sandbox
- Директивы навигации:
- - form-action, frame-ancestors
- Другие:
- - require-trusted-types-for, trusted-types, upgrade-insecure-requests

```bash
Content-Security-Policy: default-src 'self' http://example.com; connect-src 'none';
Content-Security-Policy: connect-src http://example.com/; script-src http://example.com/
```

доступ только по http

```bash
Content-Security-Policy: default-src https:
```

```html
<meta http-equiv="Content-Security-Policy" content="default-src https:" />
```

# Content-Security-Policy-Report-Only

Помогает выявить нарушения Content-Security-Policy

```bash
Content-Security-Policy-Report-Only: default-src https:; # директива
  report-uri /csp-report-url/; # куда отправлять
  report-to csp-endpoint;
```

# Content-Type

определяет медиа тип ресурса, при ответе с сервера. при post и put запросах определяет тип отправляемого контента. Content-Encoding говорит о том как декодировать

```bash
# ответы с сервера
HTTP/1.1 200
content-encoding: br
content-type: text/javascript; charset=utf-8
vary: Accept-Encoding
date: Fri, 21 Jun 2024 14:02:25 GMT
content-length: 2978

const videoPlayer=document.getElementById...

HTTP/3 200
server: nginx
date: Wed, 24 Jul 2024 16:53:02 GMT
content-type: text/css
vary: Accept-Encoding
content-encoding: br

.super-container{clear:both;max-width:100%}...
```

```bash
POST /foo HTTP/1.1
Content-Length: 68137
Content-Type: multipart/form-data; boundary=ExampleBoundaryString

--ExampleBoundaryString
Content-Disposition: form-data; name="description"

Description input value
--ExampleBoundaryString
Content-Disposition: form-data; name="myFile"; filename="foo.txt"
Content-Type: text/plain

[content of the file foo.txt chosen by the user]
---------------------------1003363413119651595289485765
```

# Cross-Origin-Embedder-Policy

управляет политикой документа и встраивает cross-origin в no-cors

```bash
Reporting-Endpoints: coep-endpoint="https://some-example.com/coep"
Cross-Origin-Embedder-Policy: require-corp; report-to="coep-endpoint"
```

# Cross-Origin-Embedder-Policy-Report-Only

конфигурация отчетов для других ресурсов

# Cross-Origin-Opener-Policy

Это позволяет веб-сайту контролировать, будет ли новый документ верхнего уровня, открытый с помощью Window.open() или при переходе на новую страницу, открыт в той же группе контекста просмотра (BCG) или в новой группе контекста просмотра.

```bash
Cross-Origin-Opener-Policy: unsafe-none # отказ
Cross-Origin-Opener-Policy: same-origin-allow-popups
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Opener-Policy: noopener-allow-popups
```

# Cross-Origin-Resource-Policy

сообщает браузеру что он должен блокировать междоменыне запросы к сайту

```bash
Cross-Origin-Resource-Policy: same-site | same-origin | cross-origin
```

# Dictionary-ID

отправляет id [словаря](../compression-dictionary-transport.md) работает в паре с Use-As-Dictionary и с активным Available-Dictionary заголовками

```bash
# ответ сервера при запросе ресурса
Use-As-Dictionary: match="/js/app.*.js", id="dictionary-12345"
# при запросе из браузера ресурса указывается Dictionary-ID: "dictionary-12345"
Accept-Encoding: gzip, br, zstd, dcb, dcz
Available-Dictionary: :pZGm1Av0IEBKARczz7exkNYsZb8LzaMrV7J32a2fFG4=:
Dictionary-ID: "dictionary-12345"
```

# ETag

является идентификатором версии ресурса. Позволяет сравнить версии ресурса. Избежать коллизий помогает <!--TODO добавить ссылку на If-Match-->If-Match, который отправляется на сервер, при изменении ресурса и он должен быть равен etag изменяемого ресурса. Если не совпадают [412 ошибка](../response-statuses.md).

Второй вариант использования - кеширования. Клиент отправляет<!--TODO добавить ссылку на If-None-Match-->If-None-Match со значением etag. Если значения совпадают, то можно использовать закешированные данные, сервер вернет 304 статус

# Expires

Содержит дату или время, когда ответ считается устаревшим в контексте кеширования, значения:

```bash
Expires: <day-name>, <day> <month> <year> <hour>:<minute>:<second> GMT
```

- 0 - всегда устаревший
- day-name: Mon, Tue, Wed, Thu, Fri, Sat, Sun (case-sensitive)
- day: "04" или "23" (2 цифры)
- month: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec
- year: "1990", "2016" (4 цифры)
- hour: 2 цифры
- minute: 2 цифры
- second: 2 цифры

!!! будет игнорироваться если есть Cache-Control max-age или s-maxage
