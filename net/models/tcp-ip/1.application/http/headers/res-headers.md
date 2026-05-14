# заголовки ответа

CH - client hint

Заголовки по типу ответ от сервера:

# Accept-CH

заголовок, который отправляет сервер, что бы получить CH заголовки от браузера, возможные значения: [Sec-CH-UA-Model](./sec-headers.md#sec-ch-ua-model), [Sec-CH-UA-Form-Factors](./sec-headers.md#sec-ch-ua-form-factors), DPR, Viewport-Width, Width - сервер требует доп информацию, браузер на каждый запрос будет отправлять Sec-CH-UA-Model, Sec-CH-UA-Form-Factors. Поведение можно реализовать в html:

```html
<meta http-equiv="Accept-CH" content="Width, Downlink, Sec-CH-UA" />
```

Пример запроса:

```bash
Accept-CH: Sec-CH-Viewport-Width, Sec-CH-Width # отправил сервер
Vary: Sec-CH-Viewport-Width, Sec-CH-Width # какие vary заголовки ждет сервер
```

# Accept-Patch

какие типы мультимедиа сервер может понимать в запросе PATCH. В случае отсутствия поддержки типа 415 Unsupported Media Type. Заголовок должен отображаться в запросах OPTIONS к ресурсу, поддерживающему метод PATCH

# Accept-Post

какие типы мультимедиа сервер может понимать в запросе Post. В случае отсутствия поддержки типа 415 Unsupported Media Type. Заголовок должен отображаться в запросах OPTIONS к ресурсу, поддерживающему метод Post

# Activate-Storage-Access

Это позволяет серверу активировать предоставленное разрешение на доступ к своим неразделенным файлам cookie в межсайтовом запросе.

```bash
Activate-Storage-Access: retry; allowed-origin="https://foo.bar"
Activate-Storage-Access: retry; allowed-origin=*
Activate-Storage-Access: load
```

заголовок предоставляет доступ к своим кукам для кросс запросов, сервер использует [Sec-Fetch-Storage-Access](./sec-headers.md#sec-fetch-storage-access) заголовок. Управляет поведением Storage Access API, который позволяет использовать сторонние куки

Принцип работы заключается в следующем:

- Браузер добавляет к запросам строку Sec-Fetch-Storage-Access: inactive, когда контекст имеет разрешение, но не активен (вместе с заголовком Origin, указывающим источник запроса)
- Если сервер получает сообщение Sec-Fetch-Storage-Access: inactive, он может ответить сообщением Activate-Storage-Access: retry; allowed-origin="<request_origin>", чтобы запросить у браузера активацию разрешения для контекста и повторную отправку запроса
- Если браузер получает запрос на повторную отправку, он активирует разрешение и отправляет запрос снова, на этот раз с параметром Sec-Fetch-Storage-Access: active и с включением файлов cookie
- Если сервер получает запрос с параметром Sec-Fetch-Storage-Access: active и содержит cookie-файлы, он отвечает версией ресурса с учетными данными. После загрузки браузером этот ресурс получает доступ к своим cookie-файлам так же, как если бы это был собственный ресурс

```bash
# запрос с Sec-Fetch-Storage-Access: inactive
GET /user/profile HTTP/1.1
Host: embedded.com
Origin: https://mysite.example
Sec-Fetch-Dest: iframe
Sec-Fetch-Site: cross-site
Sec-Fetch-Mode: navigate
Sec-Fetch-Storage-Access: inactive
Credentials-Mode: include

# ответ с Activate-Storage-Access: retry
HTTP/1.1 401 Unauthorized
Content-Type: text/html
Vary: Sec-Fetch-Storage-Access
Activate-Storage-Access: retry; allowed-origin="https://mysite.example"

# запрос с Sec-Fetch-Storage-Access: active
GET /user/profile HTTP/1.1
Host: embedded.com
Origin: https://mysite.example
Sec-Fetch-Dest: iframe
Sec-Fetch-Site: cross-site
Sec-Fetch-Mode: navigate
Sec-Fetch-Storage-Access: active
Credentials-Mode: include
Cookie: sessionid=abc123
```

# Age

сколько объект запроса находился в кеше прокси. Если значение равно 0, объект, вероятно, был получен с исходного сервера. В противном случае значение обычно вычисляется как разница между текущей датой прокси-сервера и общим заголовком Date, включенным в HTTP-ответ.

# Allow

Список дял доступных методов, должен возвращаться для 405 ошибки (Not Allowed)

# Alt-Svc

Приоритетный альтернативный ресурс

```bash
Alt-Svc: clear
Alt-Svc: <protocol-id>=<alt-authority>; ma=<max-age>
Alt-Svc: <protocol-id>=<alt-authority>; ma=<max-age>; persist=1
```

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

Если не определен заголовок CSP, то будут [использованы стандартные правила ограничения](../security/same-origin-policy.md)

директивы, определяют доступ к определенным элементам и ресурсам встроенные в страницы:

- child-src - Определяет допустимые источники для веб-воркеров и вложенных контекстов просмотра, загружаемых с помощью таких элементов, как frame и iframe. Значение для frame-src and worker-src
- connect-src - Ограничивает список URL-адресов, которые можно загружать с помощью скриптовых интерфейсов
- default-src - Служит резервным вариантом для других директив выборки
- fenced-frame-src - доступ для fencedframe.
- font-src - источники которые могут быть использованы в качестве шрифтов
- frame-src - источники для frame
- img-src - источники для изображений
- manifest-src - для манифеста
- media-src - для тегов audio/video/track
- object-src - для object и embed
- prefetch-src - для prefetch и prerendered ресурсов
- script-src - для script fallback для script-src-attr style-src-elem
- script-src-elem - для js script
- script-src-attr - для inline js событий
- style-src - для стилей
- style-src-elem - для сторонних событий стилей
- style-src-attr - для инлайн стилей
- worker-src - для Worker, SharedWorker, ServiceWorker

значения:

- nonce-nonce_value - работает в парсе с nonce атрибутами script и style тегами
- hash_algorithm-hash_value - хеш значение для script и Style
- host-source - url или ip адрес
- scheme-source - позволяет загружать ресурсы http или ws
- self - только со своего origin
- trusted-types-eval -
- unsafe-eval
- wasm-unsafe-eval
- unsafe-inline
- unsafe-hashes
- inline-speculation-rules
- strict-dynamic
- none - полная блокировка

Директивы документа:

- base-uri - редирект url, которые могут быть в base,
- sandbox - Включает изолированную среду для запрашиваемого ресурса, аналогичную атрибуту iframe

Директивы навигации:

- form-action - target значение для форм,
- frame-ancestors - Указывает допустимые родительские элементы, которые могут встраивать страницу с помощью frame, iframe, object или embed

Другие:

- require-trusted-types-for - Обеспечивает использование доверенных типов в точках внедрения XSS-атак в DOM,
- trusted-types - Используется для указания списка разрешенных политик доверенных типов. Доверенные типы позволяют приложениям блокировать механизмы внедрения DOM XSS, чтобы они принимали только неподделываемые типизированные значения вместо строк.,
- upgrade-insecure-requests - Указывает пользовательским агентам обрабатывать все небезопасные URL-адреса сайта (те, которые передаются по протоколу HTTP) так, как если бы они были заменены безопасными URL-адресами (те, которые передаются по протоколу HTTPS).

```bash
# Может быть несколько CSP заголовков
Content-Security-Policy: default-src 'self' http://example.com; connect-src 'none';
Content-Security-Policy: connect-src http://example.com/; script-src http://example.com/
```

доступ только по https

```bash
Content-Security-Policy: default-src https:
```

```html
<meta http-equiv="Content-Security-Policy" content="default-src https:" />
```

!!! обязательным является default-src для все неопределенных правил

```bash
# разрешить использовать inline-код и https-ресурсы
Content-Security-Policy: default-src https: 'unsafe-eval' 'unsafe-inline'; object-src 'none'
# пример разрешения только доменов и поддоменов
Content-Security-Policy: default-src 'self'
# источники - только исходный сервер
Content-Security-Policy: default-src 'self'
# с доверенного домена и поддомена
Content-Security-Policy: default-src 'self' *.trusted.com
# картинки из любого источника, но медиа из определенных
Content-Security-Policy: default-src 'self'; img-src *; media-src media1.com media2.com; script-src userscripts.example.com
```

# Content-Security-Policy-Report-Only

Помогает выявить нарушения Content-Security-Policy

```bash
Content-Security-Policy-Report-Only: default-src https:; # директива
  report-uri /csp-report-url/; # куда отправлять
  report-to csp-endpoint;
```

# Cross-Origin-Embedder-Policy

управляет политикой документа и встраивает cross-origin в no-cors режиме
Заголовок следует устанавливать только с одним токеном и необязательным адресом конечной точки report-to

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

# ETag

является идентификатором версии ресурса. Позволяет сравнить версии ресурса. Избежать коллизий помогает [If-Match](./req-headers.md#if-match), который отправляется на сервер, при изменении ресурса и он должен быть равен etag изменяемого ресурса. Если не совпадают [412 ошибка](../response-statuses.md).

Второй вариант использования - кеширования. Клиент отправляет [If-None-Match](./req-headers.md#if-none-match) со значением etag. Если значения совпадают, то можно использовать закешированные данные, сервер вернет 304 статус

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
