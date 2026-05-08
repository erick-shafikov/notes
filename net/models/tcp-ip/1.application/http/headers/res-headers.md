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

Определяет кто имеет доступ к ресурсу из страницы. заголовок может быть и в мета-теге

```bash
Content-Security-Policy: <policy-directive>; <policy-directive>
```

директивы:

-child-src

- connect-src
- default-src
- fenced-frame-src
- font-src
- frame-src
- img-src
- manifest-src
- media-src
- object-src
- prefetch-src
- script-src
- script-src-elem
- script-src-attr
- style-src
- style-src-elem
- style-src-attr
- worker-src

значения:

- none - полная блокировка
- self - только со своего origin
- host-source - url или ip адрес
- scheme-source - http или ws
- nonce-nonce_value,
- hash_algorithm-hash_value

Директивы документа:

- base-uri,
- sandbox

Директивы навигации:

- form-action,
- frame-ancestors

Другие:

- require-trusted-types-for,
- trusted-types,
- upgrade-insecure-requests

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

!!! обязательным является default-src для все неопределенных правил

```bash
# пример разрешения только доменов и поддоменов
Content-Security-Policy: default-src 'self'
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

# Integrity-Policy

Сообщает о том, что ресурсы должны использовать [SRI](../security/sri.md). Сигнатура:

```bash
Integrity-Policy: blocked-destinations=(<destination>),sources=(<source>),endpoints=(<endpoint>)
```

blocked-destinations:

- script
- style

sources:

- inline

endpoints - куда отправлять отчет должен быть [Reporting-Endpoints](!!!TODO link)

```bash
Reporting-Endpoints: integrity-endpoint="https://example.com/integrity", backup-integrity-endpoint="https://report-provider.example/integrity"
Integrity-Policy: blocked-destinations=(script), endpoints=(integrity-endpoint backup-integrity-endpoint)
```

# Integrity-Policy-Report-Only

Заголовок ответа HTTP Integrity-Policy-Report-Only позволяет администраторам веб-сайта сообщать о ресурсах, загружаемых пользовательским агентом, которые нарушали бы гарантии целостности подресурсов, если бы политика целостности была применена (с помощью заголовка Integrity-Policy).

```bash
Reporting-Endpoints: integrity-endpoint=https://example.com/integrity, backup-integrity-endpoint=https://report-provider.example/integrity
Integrity-Policy-Report-Only: blocked-destinations=(script), endpoints=(integrity-endpoint, backup-integrity-endpoint)
```

# Last-Modified

Содержит дату, когда ресурс был изменен. Менее предпочтительны чем [etag](#etag) но используется когда etag недоступен. Может быть использован роботами

# Location

Используется для 300-ответов или 201:

- 303 - для get с перенаправлением
- 307, 308 - для инициирующего запроса
- 301, 302 - для более старых версий

```bash
Location: /index.html
```

# Origin-Agent-Cluster

сигнализирует браузеру, что браузеру нужно распределять машинную нагрузку между одним источником

```bash
Origin-Agent-Cluster: <boolean>
```

# Preference-Applied

Добавляет информацию о том какие [Prefer](./req-headers.md#prefer) были применены

# Proxy-Authenticate

определяет метод аутентификации для доступа к ресурсу за прокси-сервером. Оно отправляется в ответе 407 Proxy Authentication Required, чтобы клиент мог идентифицировать себя перед прокси-сервером, требующим аутентификации.

```bash
Proxy-Authenticate: Basic realm="Dev", charset="UTF-8"
```

# Range

указывает что сервер должен вернуть часть ресурсов, сервер может вернуть часть ресурса с ответом 206 Partial Content. Если указанный диапазон неверный может вернуть 416 Range Not Satisfiable. если сервер не поддерживает отдачу контента по частям может вернуть сразу весь ресурс с 200. Требуется только порядковый номер байта

```bash
# Range: <unit>=<range-start>-
Range: bytes=900-
# Range: <unit>=-<range-end>
Range: bytes=-100
# Range: <unit>=<range-start>-<range-end>
Range: bytes=0-499
# Range: <unit>=<range-start>-<range-end>, …, <range-startN>-<range-endN>
Range: bytes=200-999, 2000-2499, 9500-
# Range: <unit>=-<suffix-length>
```

# Referrer-Policy

управляет тем какую информацию должен представить [Referer](./req-headers.md#referer)

```bash
Referrer-Policy: no-referrer
Referrer-Policy: no-referrer-when-downgrade
Referrer-Policy: origin
Referrer-Policy: origin-when-cross-origin
Referrer-Policy: same-origin
Referrer-Policy: strict-origin
Referrer-Policy: strict-origin-when-cross-origin
Referrer-Policy: unsafe-url
```

может быть интегрирован и в HTML

```html
<meta name="referrer" content="origin" />
<a href="http://example.com" referrerpolicy="origin">…</a>
<a href="http://example.com" rel="noreferrer">…</a>
```

# Refresh

Сообщает браузеру, что нужно обновить страницу или перенаправить. Идентично

```html
<meta http-equiv="refresh" content="..." />
```

```bash
# Refresh: <time>
Refresh: 5
# Refresh: 5; url=https://example.com/
Refresh: <time>, url=<url>
Refresh: <time>; url=<url>
```

# Reporting-Endpoints

Определяет куда отправлять отчеты при нарушении, ошибках

```bash
Reporting-Endpoints: default="https://example.com/reports"
Reporting-Endpoints: csp-endpoint="https://example.com/csp-reports"
Content-Security-Policy: default-src 'self'; report-to csp-endpoint
Reporting-Endpoints: csp-endpoint="https://example.com/csp-reports",
                     permissions-endpoint="https://example.com/permissions-policy-reports"
```

# Retry-After

Сообщает о том через сколько сделать следующий запрос:

- При 503 Service Unavailable - через сколько будет готов сервере
- При 429 Too Many Requests
- 301 Moved Permanently через сколько будет перенаправление

```bash
Retry-After: Wed, 21 Oct 2015 07:28:00 GMT
Retry-After: 120
```

# Server

описывает программное обеспечение, используемое исходным сервером, который обработал запрос и сгенерировал ответ.

```bash
Server: Apache/2.4.1 (Unix)
```

# Server-Timing

передает пользовательскому агенту одну или несколько метрик производительности цикла запрос-ответ.

```bash
Server-Timing: custom-metric;dur=123.45;desc="My custom metric"
```

# Service-Worker-Allowed

Серверы могут использовать заголовок чтобы разрешить сервис-воркеру управлять URL-адресами за пределами своего собственного каталога.

# Service-Worker-Navigation-Preload (-ff)

указывает на то, что запрос был результатом операции fetch(), выполненной во время предварительной загрузки навигации Service Worker.

# Set-Cookie

Используется для отправки cookie-файла с сервера в пользовательский агент, чтобы пользовательский агент мог позже отправить его обратно на сервер. Для отправки нескольких cookie-файлов необходимо отправить несколько заголовков Set-Cookie в одном ответе.

```bash
# установка куки
Set-Cookie: <cookie-name>=<cookie-value>
# определяет домен на который куки были отправлены
Set-Cookie: <cookie-name>=<cookie-value>; Domain=<domain-value>
# срок истечения
Set-Cookie: <cookie-name>=<cookie-value>; Expires=<date>
# запрещает fetch и XMLHttpRequest устанавливать куки
Set-Cookie: <cookie-name>=<cookie-value>; HttpOnly
# срок истечения
Set-Cookie: <cookie-name>=<cookie-value>; Max-Age=<number>
# cookie-файл следует хранить с использованием разделенного хранилища
Set-Cookie: <cookie-name>=<cookie-value>; Partitioned
# Указывает путь, который должен присутствовать в запрашиваемом URL-адресе, чтобы браузер мог отправить заголовок Cookie
Set-Cookie: <cookie-name>=<cookie-value>; Path=<path-value>
# Указывает, что cookie-файл отправляется на сервер только при запросе по схеме https: (за исключением localhost), и, следовательно, более устойчив к атакам типа «человек посередине».
Set-Cookie: <cookie-name>=<cookie-value>; Secure
# SameSite - Управляет отправкой или неотправкой cookie-файла с межсайтовыми запросами: то есть, с запросами, исходящими с другого сайта, включая указанную схему, и с сайта, установившего cookie-файл.
# Файл cookie отправляется только для запросов, поступающих с того же сайта, который установил этот файл co
Set-Cookie: <cookie-name>=<cookie-value>; SameSite=Strict
# исключи запросы, выполненные с использованием API fetch(), или запросы к подресурсам из элементов <img> или <script>, или навигацию внутри элементов <iframe>. Запросы, выполняемые, когда пользователь переходит по ссылке в контексте просмотра верхнего уровня с одного сайта на другой, или при назначении объекта document.location, или при отправке формы.
Set-Cookie: <cookie-name>=<cookie-value>; SameSite=Lax
# Отправляйте cookie-файл как с межсайтовыми, так и с внутрисайтовыми запросами. При использовании этого значения также необходимо установить атрибут Secure
Set-Cookie: <cookie-name>=<cookie-value>; SameSite=None; Secure

// Multiple attributes are also possible, for example:
Set-Cookie: <cookie-name>=<cookie-value>; Domain=<domain-value>; Secure; HttpOnly
```

Некоторые имена cookie содержат префиксы, которые накладывают определенные ограничения на атрибуты cookie при поддержке пользовательских агентов. Все префиксы cookie начинаются с двойного подчеркивания (\_\_) и заканчиваются дефисом (-). Определены следующие префиксы.

```bash
Set-Cookie: __Secure-ID=123; Secure; Domain=example.com
Set-Cookie: __Host-ID=123; Secure; Path=/
Set-Cookie: __Secure-id=1
Set-Cookie: __Host-id=1; Secure
Set-Cookie: __Host-id=1; Secure; Path=/; Domain=example.com
Set-Cookie: __Http-ID=123; Secure; Domain=example.com
Set-Cookie: __Host-Http-ID=123; Secure; Path=/
```

# Set-Login

отправляется федеративным поставщиком идентификации (IdP) для установки статуса авторизации и указывает, вошли ли какие-либо пользователи в систему IdP в текущем браузере или нет.

# SourceMap

предоставляет расположение source map ресурса

```bash
HTTP/1.1 200 OK
Content-Type: text/javascript
SourceMap: /path/to/file.js.map

<optimized-javascript>
```

# Strict-Transport-Security

сообщает браузерам, что доступ к хосту следует осуществлять только по протоколу HTTPS, и что любые будущие попытки доступа к нему по протоколу HTTP должны автоматически переключаться на HTTPS.

# Timing-Allow-Origin

указывает, каким источникам разрешено видеть значения атрибутов, полученных с помощью функций API Resource Timing, которые в противном случае были бы указаны как нулевые из-за ограничений на междоменные запросы.

# Vary

Описывает информацию о том что повлияло на формирование текущего сообщения. Используется для кеширования. Если ничего не поменялось 304 Not Modified и ответ "default"

```bash
# Vary: <header-name>, …, <header-nameN>
Vary: * # не подлежит кешированию
```

# WWW-Authenticate

Содержит информацию о методах HTTP-аутентификации (или запросах аутентификации), которые могут быть использованы для получения доступа к конкретному ресурсу. При неверной аутентификации сервер должен вернуть 401 Unauthorized

в свою очередь клиент отвечает заголовком [Authorization](./req-headers.md#authorization)

```bash


# WWW-Authenticate: <challenge>
# challenge = <auth-scheme> <auth-param>, …, <auth-paramN>
# challenge = <auth-scheme> <token68>
# <auth-scheme> - Basic, Digest, Negotiate and AWS4-HMAC-SHA256
# <auth-param> - <realm> - realm="строка которая описывает защищенную область"
# <token68> - токен для схемы
WWW-Authenticate: Basic realm="Dev", charset="UTF-8"
```

Варианты auth-scheme и их структура:

- Basic структура:
- - realm
- - charset="UTF-8"
- Digest:
- - realm
- - domain
- - nonce
- - opaque
- - stale
- - algorithm
- - qop
- - charset="UTF-8"
- - userhash - "true" "false"
- HTTP Origin-Bound Authentication (HOBA):
- - challenge - len:value
- - max-age
- - realm

```bash
# Digest
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Digest
    realm="http-auth@example.org",
    qop="auth, auth-int",
    algorithm=SHA-256,
    nonce="7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v",
    opaque="FQhe/qaU925kfnzjCev0ciny7QMkPqMAFRtzCUYo5tdS"
WWW-Authenticate: Digest
    realm="http-auth@example.org",
    qop="auth, auth-int",
    algorithm=MD5,
    nonce="7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v",
    opaque="FQhe/qaU925kfnzjCev0ciny7QMkPqMAFRtzCUYo5tdS"

Authorization: Digest username="Mufasa",
    realm="http-auth@example.org",
    uri="/dir/index.html",
    algorithm=MD5,
    nonce="7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v",
    nc=00000001,
    cnonce="f2/wE4q74E6zIJEtWaHKaf5wv/H5QzzpXusqGemxURZJ",
    qop=auth,
    response="8ca523f5e9506fed4657c9700eebdbec",
    opaque="FQhe/qaU925kfnzjCev0ciny7QMkPqMAFRtzCUYo5tdS"
```

HOBA

```bash
HTTP/1.1 401 Unauthorized
WWW-Authenticate: HOBA max-age="180", challenge="16:MTEyMzEyMzEyMw==1:028:https://www.example.com:8080:3:MTI48:NjgxNDdjOTctNDYxYi00MzEwLWJlOWItNGM3MDcyMzdhYjUz"

Authorization: 123.16:MTEyMzEyMzEyMw==1:028:https://www.example.com:8080:3:MTI48:NjgxNDdjOTctNDYxYi00MzEwLWJlOWItNGM3MDcyMzdhYjUz.1123123123.<signature-of-challenge>
```
