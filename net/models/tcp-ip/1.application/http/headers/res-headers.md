# заголовки ответа

CH - client hint

# Accept-CH

Сервер запрашивает у браузера определённые Client Hint заголовки, которые браузер будет прикладывать к последующим запросам к этому хосту. Браузер запоминает список на всё время сессии.

Возможные значения: [Sec-CH-UA-Model](./sec-headers.md#sec-ch-ua-model), [Sec-CH-UA-Form-Factors](./sec-headers.md#sec-ch-ua-form-factors), [Sec-CH-Viewport-Width](./sec-headers.md#sec-ch-viewport-width), [Sec-CH-Width](./sec-headers.md#sec-ch-width), DPR, Viewport-Width, Width, Downlink, RTT, Device-Memory и др.

```bash
Accept-CH: Sec-CH-UA-Model, Sec-CH-Viewport-Width
Vary: Sec-CH-UA-Model, Sec-CH-Viewport-Width  # кеш учитывает эти заголовки
```

```bash
# 1. первый запрос — браузер ничего не знает
GET /page HTTP/1.1
Host: example.com

# 2. сервер сообщает какие CH нужны
HTTP/1.1 200 OK
Accept-CH: Sec-CH-UA-Model, Sec-CH-Viewport-Width
Vary: Sec-CH-UA-Model, Sec-CH-Viewport-Width

# 3. следующий запрос — браузер прикладывает запрошенные CH
GET /image.jpg HTTP/1.1
Host: example.com
Sec-CH-UA-Model: "Pixel 7"
Sec-CH-Viewport-Width: "390"
```

Можно задать через HTML:

```html
<meta http-equiv="Accept-CH" content="Width, Downlink, Sec-CH-UA" />
```

# Accept-Patch

Сообщает какие медиатипы сервер принимает в теле PATCH-запроса. Отображается в ответах на OPTIONS-запрос к ресурсу, поддерживающему PATCH. Если клиент отправляет неподдерживаемый тип — 415 Unsupported Media Type.

```bash
Accept-Patch: application/json
Accept-Patch: application/merge-patch+json
Accept-Patch: text/plain; charset=utf-8
```

```bash
# клиент проверяет поддерживаемые форматы
OPTIONS /api/user/123 HTTP/1.1
Host: api.example.com

HTTP/1.1 204 No Content
Allow: GET, POST, PATCH, DELETE, OPTIONS
Accept-Patch: application/merge-patch+json, application/json
```

```bash
# клиент отправляет PATCH с неподдерживаемым типом
PATCH /api/user/123 HTTP/1.1
Host: api.example.com
Content-Type: text/xml

<user><name>Alice</name></user>

HTTP/1.1 415 Unsupported Media Type
Accept-Patch: application/merge-patch+json
```

# Accept-Post

Сообщает какие медиатипы сервер принимает в теле POST-запроса. Отображается в ответах на OPTIONS. Если тип не поддерживается — 415 Unsupported Media Type.

```bash
Accept-Post: application/json
Accept-Post: multipart/form-data
```

```bash
OPTIONS /api/upload HTTP/1.1
Host: api.example.com

HTTP/1.1 204 No Content
Allow: GET, POST, OPTIONS
Accept-Post: multipart/form-data, application/octet-stream
```

# Activate-Storage-Access

Позволяет серверу активировать разрешение на доступ к неразделённым куки в межсайтовом запросе (Storage Access API). Сервер использует заголовок [Sec-Fetch-Storage-Access](./sec-headers.md#sec-fetch-storage-access) для проверки текущего статуса доступа.

```bash
Activate-Storage-Access: retry; allowed-origin="https://foo.bar"
Activate-Storage-Access: retry; allowed-origin=*
Activate-Storage-Access: load
```

Принцип работы:

- Браузер добавляет `Sec-Fetch-Storage-Access: inactive` если разрешение есть, но не активировано
- Сервер отвечает `Activate-Storage-Access: retry; allowed-origin=...` — браузер активирует разрешение и повторяет запрос
- Повторный запрос идёт с `Sec-Fetch-Storage-Access: active` и включёнными куки

```bash
# 1. браузер — разрешение есть, но не активировано
GET /user/profile HTTP/1.1
Host: embedded.com
Origin: https://mysite.example
Sec-Fetch-Storage-Access: inactive

# 2. сервер просит активировать и повторить
HTTP/1.1 401 Unauthorized
Vary: Sec-Fetch-Storage-Access
Activate-Storage-Access: retry; allowed-origin="https://mysite.example"

# 3. браузер активирует и повторяет запрос с куки
GET /user/profile HTTP/1.1
Host: embedded.com
Origin: https://mysite.example
Sec-Fetch-Storage-Access: active
Cookie: sessionid=abc123

HTTP/1.1 200 OK
Content-Type: application/json

{"user": "alice"}
```

# Age

Сколько секунд объект находился в кеше прокси. Если `0` — вероятно, получен напрямую с исходного сервера. Вычисляется как разница между текущим временем прокси и заголовком `Date` в ответе.

```bash
Age: <seconds>

Age: 0     # только что получен с исходного сервера
Age: 3600  # 1 час в кеше
```

```bash
GET /image.png HTTP/1.1
Host: example.com

# прокси отдаёт из кеша — объект пролежал 24 минуты
HTTP/1.1 200 OK
Date: Mon, 12 May 2025 10:00:00 GMT
Cache-Control: public, max-age=3600
Age: 1440
Content-Type: image/png
```

# Allow

Перечисляет HTTP-методы, допустимые для данного ресурса. Обязателен в ответе 405 Method Not Allowed. Также возвращается в ответах на OPTIONS.

```bash
Allow: GET, POST, HEAD
Allow: GET, HEAD, PUT
```

```bash
# клиент пытается удалить ресурс без разрешения
DELETE /api/article/42 HTTP/1.1
Host: api.example.com

HTTP/1.1 405 Method Not Allowed
Allow: GET, POST, HEAD, OPTIONS
Content-Type: application/json

{"error": "DELETE not allowed on this resource"}
```

```bash
# OPTIONS — клиент спрашивает что разрешено
OPTIONS /api/article/42 HTTP/1.1
Host: api.example.com

HTTP/1.1 204 No Content
Allow: GET, POST, HEAD, OPTIONS
```

# Alt-Svc

Рекламирует альтернативный сервис для этого же ресурса — другой хост, порт или протокол. Используется для миграции на HTTP/3 (QUIC), перехода на другой CDN-узел. Клиент может использовать альтернативу при следующем запросе.

```bash
Alt-Svc: clear                                        # отзыв всех ранее объявленных альтернатив
Alt-Svc: <protocol-id>=<alt-authority>; ma=<max-age>
Alt-Svc: <protocol-id>=<alt-authority>; ma=<max-age>; persist=1  # сохранять через перезапуск браузера

Alt-Svc: h3=":443"; ma=86400           # HTTP/3 на том же хосте, порт 443, кешировать сутки
Alt-Svc: h3=":443"; ma=86400, h2=":443"; ma=86400  # несколько вариантов
```

```bash
# сервер объявляет что поддерживает HTTP/3
GET /page HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Alt-Svc: h3=":443"; ma=86400
Content-Type: text/html

# браузер запоминает и при следующем запросе использует HTTP/3
# (новое соединение по QUIC на порт 443)
```

# Clear-Site-Data

Инструктирует браузер очистить все данные, связанные с исходным сайтом. Применяется при выходе из системы, смене аккаунта, обновлении версии. Браузер очищает только данные текущего origin.

```bash
Clear-Site-Data: "cache"              # HTTP-кеш
Clear-Site-Data: "cookies"            # все куки
Clear-Site-Data: "storage"            # localStorage, sessionStorage, IndexedDB, etc.
Clear-Site-Data: "executionContexts"  # Workers, Service Workers
Clear-Site-Data: "prefetchCache"      # prefetch-кеш
Clear-Site-Data: "prerenderCache"     # prerender-кеш
Clear-Site-Data: "*"                  # всё перечисленное

# несколько значений
Clear-Site-Data: "cache", "cookies", "storage"
```

```bash
# выход из системы — сервер чистит куки и кеш
POST /logout HTTP/1.1
Host: example.com
Cookie: session=abc123

HTTP/1.1 200 OK
Clear-Site-Data: "cache", "cookies"
Content-Type: application/json

{"status": "logged out"}
```

# Content-Security-Policy

Определяет политику безопасности содержимого страницы — ограничивает источники, из которых браузер может загружать ресурсы. Если заголовок не задан, применяются стандартные правила браузера. Можно задать несколько CSP-заголовков — будет применяться наиболее строгий.

```bash
Content-Security-Policy: <policy-directive>; <policy-directive>
```

**Директивы выборки** (из каких источников можно загружать):

```bash
default-src   # fallback для всех директив, которые не заданы явно
script-src    # JS-скрипты (fallback для script-src-elem и script-src-attr)
script-src-elem  # <script src="...">
script-src-attr  # inline-обработчики onclick="..."
style-src     # CSS-стили
style-src-elem   # <link rel="stylesheet">, <style>
style-src-attr   # атрибут style="..."
img-src       # изображения
font-src      # шрифты
connect-src   # fetch, XMLHttpRequest, WebSocket
media-src     # <audio>, <video>
object-src    # <object>, <embed>
frame-src     # <iframe>
child-src     # <iframe> + Workers
worker-src    # Worker, SharedWorker, ServiceWorker
manifest-src  # Web App Manifest
prefetch-src  # prefetch и prerender
fenced-frame-src  # <fencedframe>
```

**Значения источников:**

```bash
'none'              # блокировать всё
'self'              # только текущий origin
'unsafe-inline'     # разрешить inline-скрипты и стили (ослабляет защиту)
'unsafe-eval'       # разрешить eval()
'strict-dynamic'    # доверять скриптам, которым доверяет уже доверенный скрипт
https:              # любой HTTPS-источник
'nonce-<base64>'    # разрешить конкретный тег с совпадающим nonce-атрибутом
'sha256-<base64>'   # разрешить inline-контент с совпадающим хешем
*.trusted.com       # поддомены конкретного домена
```

**Директивы документа:**

```bash
base-uri     # допустимые URL для <base>
sandbox      # изолированная среда (как у iframe)
```

**Директивы навигации:**

```bash
form-action      # куда можно отправлять формы
frame-ancestors  # кто может встраивать страницу через <iframe>
```

**Отчётность:**

```bash
upgrade-insecure-requests   # автоматически заменять http:// на https://
require-trusted-types-for   # DOM-based XSS защита
report-to <endpoint>        # куда слать отчёты о нарушениях
```

```bash
# строгая политика: только свой origin, без inline
Content-Security-Policy: default-src 'self'; object-src 'none'; base-uri 'none'

# только HTTPS-ресурсы
Content-Security-Policy: default-src https:

# скрипты только со своего домена и CDN, изображения откуда угодно
Content-Security-Policy: default-src 'self'; img-src *; script-src 'self' cdn.example.com

# nonce для конкретного скрипта
Content-Security-Policy: script-src 'nonce-rAnd0m123'
# → в HTML: <script nonce="rAnd0m123">...</script>

# несколько CSP (применяется наиболее строгий)
Content-Security-Policy: default-src 'self' http://example.com; connect-src 'none';
Content-Security-Policy: connect-src http://example.com/; script-src http://example.com/
```

```html
<meta http-equiv="Content-Security-Policy" content="default-src https:" />
```

# Content-Security-Policy-Report-Only

Работает как [Content-Security-Policy](#content-security-policy), но не блокирует ресурсы, а только отправляет отчёты о нарушениях. Используется для тестирования политики перед применением.

```bash
Content-Security-Policy-Report-Only: <policy>; report-to <endpoint>
```

```bash
# сервер тестирует политику — нарушения логируются, но не блокируются
HTTP/1.1 200 OK
Reporting-Endpoints: csp-violations="https://example.com/csp-report"
Content-Security-Policy-Report-Only: default-src 'self'; script-src 'self'; report-to csp-violations

# браузер обнаружил нарушение (сторонний скрипт) и отправляет отчёт
POST /csp-report HTTP/1.1
Host: example.com
Content-Type: application/reports+json

[{
  "type": "csp-violation",
  "body": {
    "documentURL": "https://example.com/page",
    "blockedURL": "https://evil.com/tracker.js",
    "effectiveDirective": "script-src-elem",
    "originalPolicy": "default-src 'self'; script-src 'self'"
  }
}]
```

# Cross-Origin-Embedder-Policy

Управляет загрузкой ресурсов из сторонних источников. Требуется вместе с [Cross-Origin-Opener-Policy: same-origin](#cross-origin-opener-policy) для включения функций, требующих изоляции (например, `SharedArrayBuffer`, `performance.measureUserAgentSpecificMemory()`).

```bash
Cross-Origin-Embedder-Policy: unsafe-none        # без ограничений (по умолчанию)
Cross-Origin-Embedder-Policy: require-corp       # все ресурсы должны иметь CORP/CORS
Cross-Origin-Embedder-Policy: credentialless     # no-cors запросы без учётных данных
```

```bash
# включение изоляции для SharedArrayBuffer
HTTP/1.1 200 OK
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Content-Type: text/html

# JS теперь имеет доступ к SharedArrayBuffer и высокоточному таймеру
```

```bash
# с репортингом нарушений
Reporting-Endpoints: coep-endpoint="https://example.com/coep-report"
Cross-Origin-Embedder-Policy: require-corp; report-to="coep-endpoint"
```

# Cross-Origin-Embedder-Policy-Report-Only

Тестовый режим [Cross-Origin-Embedder-Policy](#cross-origin-embedder-policy) — нарушения политики отправляются в отчётах, но ресурсы не блокируются.

```bash
Reporting-Endpoints: coep-endpoint="https://example.com/coep-report"
Cross-Origin-Embedder-Policy-Report-Only: require-corp; report-to="coep-endpoint"
```

# Cross-Origin-Opener-Policy

Управляет тем, в какой группе контекста просмотра (BCG) откроется новый документ верхнего уровня. Влияет на доступность `window.opener` и позволяет изолировать документ от открывших его окон.

```bash
Cross-Origin-Opener-Policy: unsafe-none              # без изоляции (по умолчанию)
Cross-Origin-Opener-Policy: same-origin-allow-popups # изоляция, но сохраняет opener для своих попапов
Cross-Origin-Opener-Policy: same-origin              # полная изоляция — только same-origin в одной BCG
Cross-Origin-Opener-Policy: noopener-allow-popups    # без opener, но попапы разрешены
```

```bash
# страница изолирована — window.opener недоступен с других origin
HTTP/1.1 200 OK
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Content-Type: text/html
```

# Cross-Origin-Resource-Policy

Сообщает браузеру заблокировать cross-origin и cross-site запросы к данному ресурсу без явного CORS-разрешения. Защищает ресурсы от Spectre-подобных атак через чтение кросс-доменных ответов из кеша.

```bash
Cross-Origin-Resource-Policy: same-site    # только с того же сайта (scheme + registrable domain)
Cross-Origin-Resource-Policy: same-origin  # только с того же origin (scheme + host + port)
Cross-Origin-Resource-Policy: cross-origin # доступен с любого origin (как нет ограничений)
```

```bash
# API, доступный только с того же сайта
HTTP/1.1 200 OK
Cross-Origin-Resource-Policy: same-site
Content-Type: application/json

{"data": "sensitive"}
```

# ETag

Идентификатор версии ресурса, вычисляемый сервером (хеш содержимого, timestamp, revision). Используется в двух сценариях: **кеширование** и **предотвращение коллизий**.

```bash
ETag: "abc123"        # сильный ETag (exact match)
ETag: W/"abc123"      # слабый ETag (W/ — семантически эквивалентный, не побайтово)
```

**Кеширование** — клиент проверяет актуальность кеша через [If-None-Match](./req-headers.md#if-none-match):

```bash
# 1. первый запрос
GET /data.json HTTP/1.1
Host: api.example.com

HTTP/1.1 200 OK
ETag: "v3-abc123"
Cache-Control: max-age=60
Content-Type: application/json

{"users": [...]}

# 2. кеш устарел — клиент ревалидирует
GET /data.json HTTP/1.1
Host: api.example.com
If-None-Match: "v3-abc123"

# 3а. данные не изменились — тело не передаётся
HTTP/1.1 304 Not Modified
ETag: "v3-abc123"

# 3б. данные изменились — новый ответ
HTTP/1.1 200 OK
ETag: "v4-def456"
Content-Type: application/json

{"users": [...updated...]}
```

**Предотвращение коллизий** — клиент передаёт ETag при изменении через [If-Match](./req-headers.md#if-match):

```bash
# 1. получить ресурс с ETag
GET /api/article/42 HTTP/1.1

HTTP/1.1 200 OK
ETag: "v1-xyz"

{"title": "Old Title"}

# 2. обновить только если не изменился
PUT /api/article/42 HTTP/1.1
If-Match: "v1-xyz"
Content-Type: application/json

{"title": "New Title"}

# 2а. версии совпадают — обновлено
HTTP/1.1 200 OK
ETag: "v2-qrs"

# 2б. кто-то уже изменил ресурс — конфликт
HTTP/1.1 412 Precondition Failed
```

# Expires

Дата/время, когда ответ считается устаревшим для целей кеширования. Устаревший формат HTTP/1.0, заменён директивой `max-age` в [Cache-Control](./req-res-headers.md#cache-control). Если присутствует `Cache-Control: max-age` или `s-maxage` — заголовок `Expires` игнорируется.

```bash
Expires: <day-name>, <day> <month> <year> <hour>:<minute>:<second> GMT

Expires: Wed, 21 Oct 2025 07:28:00 GMT
Expires: 0   # ресурс считается уже устаревшим (отключить кеш)
```

```bash
GET /logo.png HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Expires: Thu, 12 Jun 2025 00:00:00 GMT
Content-Type: image/png
Content-Length: 12345
```

# Integrity-Policy

Требует, чтобы ресурсы указанных типов были загружены с атрибутом [integrity (SRI)](../security/sri.md). Если ресурс загружается без integrity — блокируется и отправляется отчёт.

```bash
Integrity-Policy: blocked-destinations=(<destination>), sources=(<source>), endpoints=(<endpoint>)
```

- `blocked-destinations` — для каких типов ресурсов требовать integrity: `script`, `style`
- `sources` — какие источники требуют integrity: `inline`
- `endpoints` — куда слать отчёты (имена из [Reporting-Endpoints](#reporting-endpoints))

```bash
Reporting-Endpoints: integrity-endpoint="https://example.com/integrity"
Integrity-Policy: blocked-destinations=(script style), endpoints=(integrity-endpoint)
```

```bash
# скрипт без integrity — блокируется
HTTP/1.1 200 OK
Reporting-Endpoints: integrity-ep="https://example.com/integrity-report"
Integrity-Policy: blocked-destinations=(script), endpoints=(integrity-ep)
Content-Type: text/html

# <script src="https://cdn.example.com/lib.js"></script> — будет заблокирован
# <script src="https://cdn.example.com/lib.js" integrity="sha384-..."></script> — пройдёт
```

# Integrity-Policy-Report-Only

Тестовый режим [Integrity-Policy](#integrity-policy) — нарушения фиксируются в отчётах, но ресурсы не блокируются.

```bash
Reporting-Endpoints: integrity-endpoint="https://example.com/integrity"
Integrity-Policy-Report-Only: blocked-destinations=(script), endpoints=(integrity-endpoint)
```

# Last-Modified

Дата последнего изменения ресурса. Менее надёжна, чем [ETag](#etag) (не учитывает изменения без изменения времени), но используется как fallback. Клиент может использовать в условных запросах через [If-Modified-Since](./req-headers.md#if-modified-since) и [If-Unmodified-Since](./req-headers.md#if-unmodified-since).

```bash
Last-Modified: <day-name>, <day> <month> <year> <hour>:<minute>:<second> GMT

Last-Modified: Wed, 21 Oct 2024 07:28:00 GMT
```

```bash
# 1. первый запрос
GET /page.html HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Last-Modified: Wed, 21 Oct 2024 07:28:00 GMT
Content-Type: text/html

(html body)

# 2. повторный запрос — клиент спрашивает изменилось ли
GET /page.html HTTP/1.1
Host: example.com
If-Modified-Since: Wed, 21 Oct 2024 07:28:00 GMT

# 3а. не изменилось
HTTP/1.1 304 Not Modified

# 3б. изменилось
HTTP/1.1 200 OK
Last-Modified: Thu, 12 May 2025 10:00:00 GMT
Content-Type: text/html

(новый html body)
```

# Location

Указывает URL для перенаправления или URL созданного ресурса. Семантика зависит от кода ответа:

- **201 Created** — URL созданного ресурса
- **301, 302** — постоянное/временное перенаправление (метод может измениться на GET)
- **303 See Other** — перенаправляет на GET-ресурс (после POST/PUT/DELETE)
- **307, 308** — перенаправление с сохранением метода запроса

```bash
Location: /index.html
Location: https://other.example.com/page
```

```bash
# 201 — ресурс создан, Location указывает где его найти
POST /api/users HTTP/1.1
Content-Type: application/json

{"name": "Alice"}

HTTP/1.1 201 Created
Location: /api/users/42
Content-Type: application/json

{"id": 42, "name": "Alice"}
```

```bash
# 301 — постоянное перенаправление
GET /old-page HTTP/1.1
Host: example.com

HTTP/1.1 301 Moved Permanently
Location: https://example.com/new-page
```

```bash
# 303 — после POST перенаправить на GET (PRG-паттерн)
POST /checkout HTTP/1.1
Content-Type: application/x-www-form-urlencoded

item=42&qty=1

HTTP/1.1 303 See Other
Location: /order/confirmation/9921

# браузер делает GET /order/confirmation/9921
```

# Origin-Agent-Cluster

Сигнализирует браузеру изолировать текущий документ в отдельном кластере агентов на уровне origin, а не site. Это позволяет ОС выделять документу отдельный процесс, что повышает изоляцию (защита от Spectre) но ограничивает `document.domain`.

```bash
Origin-Agent-Cluster: ?1  # запросить изоляцию по origin (Structured Field boolean)
Origin-Agent-Cluster: ?0  # отказаться от изоляции
```

```bash
HTTP/1.1 200 OK
Origin-Agent-Cluster: ?1
Content-Type: text/html
```

# Preference-Applied

Сообщает клиенту какие из запрошенных [Prefer](./req-headers.md#prefer) предпочтений были применены сервером.

```bash
# клиент просит вернуть только заголовки
GET /api/resource HTTP/1.1
Prefer: return=minimal

HTTP/1.1 200 OK
Preference-Applied: return=minimal
Content-Type: application/json

{}
```

# Proxy-Authenticate

Сообщает какой метод аутентификации нужен для доступа через прокси-сервер. Отправляется в ответе **407 Proxy Authentication Required**. Клиент отвечает заголовком [Proxy-Authorization](./req-headers.md#proxy-authorization).

```bash
Proxy-Authenticate: Basic realm="Corporate Proxy"
Proxy-Authenticate: Digest realm="proxy.example.com", nonce="abc123"
```

```bash
# 1. клиент запрашивает ресурс через прокси
GET https://external.example.com/api HTTP/1.1
Host: external.example.com

# 2. прокси требует аутентификацию
HTTP/1.1 407 Proxy Authentication Required
Proxy-Authenticate: Basic realm="Corporate Proxy", charset="UTF-8"

# 3. клиент повторяет с учётными данными
GET https://external.example.com/api HTTP/1.1
Host: external.example.com
Proxy-Authorization: Basic dXNlcjpwYXNz

HTTP/1.1 200 OK
Content-Type: application/json

{"data": "..."}
```

# Referrer-Policy

Управляет тем, какую информацию из URL браузер передаёт в заголовке [Referer](./req-headers.md#referer) при навигации и загрузке ресурсов.

```bash
Referrer-Policy: no-referrer                    # никогда не отправлять Referer
Referrer-Policy: no-referrer-when-downgrade     # не отправлять при переходе HTTPS→HTTP
Referrer-Policy: origin                         # только origin (scheme+host+port), без пути
Referrer-Policy: origin-when-cross-origin       # полный URL внутри origin, только origin — для cross-origin
Referrer-Policy: same-origin                    # полный URL только для same-origin, cross-origin — без referer
Referrer-Policy: strict-origin                  # только origin, не отправлять при HTTPS→HTTP
Referrer-Policy: strict-origin-when-cross-origin  # (по умолчанию) полный URL same-origin, origin — cross-origin, не отправлять HTTPS→HTTP
Referrer-Policy: unsafe-url                     # всегда полный URL включая путь и query
```

```bash
HTTP/1.1 200 OK
Referrer-Policy: strict-origin-when-cross-origin
Content-Type: text/html
```

Можно задать в HTML:

```html
<meta name="referrer" content="origin" />
<a href="https://external.com" referrerpolicy="no-referrer">Link</a>
<a href="https://external.com" rel="noreferrer">Link</a>
```

# Refresh

Инструктирует браузер автоматически перезагрузить страницу через заданное количество секунд или перейти по URL.

```bash
Refresh: <seconds>
Refresh: <seconds>; url=<url>
```

```bash
# перезагрузить через 5 секунд
HTTP/1.1 200 OK
Refresh: 5
Content-Type: text/html

# перейти на другую страницу через 3 секунды
HTTP/1.1 200 OK
Refresh: 3; url=https://example.com/new-page
Content-Type: text/html

<p>Вы будете перенаправлены через 3 секунды...</p>
```

Эквивалент в HTML:

```html
<meta http-equiv="refresh" content="5; url=https://example.com/" />
```

# Reporting-Endpoints

Определяет именованные эндпоинты для отправки отчётов о нарушениях CSP, COOP, COEP, Network Error Logging и др. Имена эндпоинтов используются в директивах `report-to` других заголовков. Замена устаревшего [Report-To](./deprecated.md#report-to-res).

```bash
Reporting-Endpoints: <name>="<url>"
Reporting-Endpoints: default="https://example.com/reports"
Reporting-Endpoints: csp-endpoint="https://example.com/csp", nel-endpoint="https://example.com/nel"
```

```bash
HTTP/1.1 200 OK
Reporting-Endpoints: csp-endpoint="https://example.com/csp-reports",
                     permissions-endpoint="https://example.com/permissions-reports"
Content-Security-Policy: default-src 'self'; report-to csp-endpoint
Permissions-Policy: geolocation=(); report-to=permissions-endpoint
```

# Retry-After

Сообщает клиенту через сколько времени повторить запрос. Используется при:

- **503 Service Unavailable** — сервер временно недоступен
- **429 Too Many Requests** — превышен rate limit
- **301 Moved Permanently** — через сколько будет доступен новый URL

```bash
Retry-After: <seconds>
Retry-After: <http-date>

Retry-After: 120
Retry-After: Wed, 21 Oct 2025 07:28:00 GMT
```

```bash
# rate limiting — превышен лимит запросов
GET /api/data HTTP/1.1
Host: api.example.com

HTTP/1.1 429 Too Many Requests
Retry-After: 60
Content-Type: application/json

{"error": "Rate limit exceeded. Try again in 60 seconds."}
```

```bash
# технические работы
HTTP/1.1 503 Service Unavailable
Retry-After: Wed, 21 May 2025 06:00:00 GMT
Content-Type: text/html

<p>Site is under maintenance.</p>
```

# Server

Описывает серверное ПО, обработавшее запрос. Рекомендуется не раскрывать точную версию в целях безопасности.

```bash
Server: Apache
Server: Apache/2.4.1 (Unix)
Server: nginx/1.25.3
Server: cloudflare
```

```bash
HTTP/1.1 200 OK
Server: nginx
Content-Type: text/html
```

# Server-Timing

Передаёт клиенту метрики производительности серверной обработки запроса. Доступны через Performance API в браузере (`PerformanceServerTiming`). Может передавать несколько метрик через запятую.

```bash
# Server-Timing: <name>; dur=<ms>; desc="<description>"
Server-Timing: db; dur=53; desc="Database query"
Server-Timing: cache; dur=1; desc="Cache lookup"
Server-Timing: app; dur=47.2; desc="Application"
Server-Timing: db;dur=53, cache;dur=1, app;dur=47.2
```

```bash
GET /api/products HTTP/1.1
Host: api.example.com

HTTP/1.1 200 OK
Content-Type: application/json
Server-Timing: db;dur=53;desc="DB query", cache;dur=1;desc="Cache hit", total;dur=72
Content-Length: 4096

[{"id": 1, ...}]
```

```js
// в браузере
const entries = performance.getEntriesByType("resource");
entries[0].serverTiming.forEach((t) => console.log(t.name, t.duration));
// db 53
// cache 1
// total 72
```

# Service-Worker-Allowed

Позволяет Service Worker перехватывать запросы за пределами собственного каталога. По умолчанию Service Worker управляет только URL в своём пути и ниже. Заголовок расширяет область до указанного пути.

```bash
Service-Worker-Allowed: /
Service-Worker-Allowed: /app/
```

```bash
# Service Worker зарегистрирован по /app/sw.js
# без заголовка управляет только /app/**
GET /app/sw.js HTTP/1.1

# сервер разрешает управлять всем сайтом
HTTP/1.1 200 OK
Content-Type: text/javascript
Service-Worker-Allowed: /

(код Service Worker)
```

# Service-Worker-Navigation-Preload

Указывает, что запрос был инициирован механизмом Navigation Preload в Service Worker. Service Worker запускается параллельно с предварительной загрузкой навигационного запроса, ускоряя первую загрузку страницы. Значение заголовка устанавливается через `registration.navigationPreload.setHeaderValue()`.

```bash
Service-Worker-Navigation-Preload: true      # значение по умолчанию
Service-Worker-Navigation-Preload: <custom>  # произвольное значение для разных стратегий
```

```bash
# браузер отправляет preload-запрос пока запускается SW
GET /page HTTP/1.1
Host: example.com
Service-Worker-Navigation-Preload: true

HTTP/1.1 200 OK
Content-Type: text/html
Vary: Service-Worker-Navigation-Preload

(html body)
```

# Set-Cookie

Устанавливает куки на клиенте. Для нескольких куки — несколько заголовков `Set-Cookie` в одном ответе. Клиент отправляет куки обратно через заголовок `Cookie`.

```bash
Set-Cookie: <name>=<value>
Set-Cookie: <name>=<value>; Domain=<domain>   # домен, которому принадлежит куки
Set-Cookie: <name>=<value>; Path=<path>        # путь, при котором куки отправляется
Set-Cookie: <name>=<value>; Expires=<date>     # дата истечения (абсолютная)
Set-Cookie: <name>=<value>; Max-Age=<seconds>  # срок жизни в секундах (приоритетнее Expires)
Set-Cookie: <name>=<value>; Secure             # только по HTTPS
Set-Cookie: <name>=<value>; HttpOnly           # недоступен JS (document.cookie)
Set-Cookie: <name>=<value>; Partitioned        # разделённое хранилище (CHIPS)
Set-Cookie: <name>=<value>; SameSite=Strict    # только same-site запросы
Set-Cookie: <name>=<value>; SameSite=Lax       # same-site + top-level navigation
Set-Cookie: <name>=<value>; SameSite=None; Secure  # cross-site (требует Secure)
```

**SameSite:**

- `Strict` — только запросы с того же сайта (никакой навигации с внешних ссылок)
- `Lax` — same-site + навигация по ссылкам верхнего уровня (по умолчанию)
- `None` — любые запросы, включая cross-site (только с `Secure`)

**Префиксы имён:**

```bash
Set-Cookie: __Secure-ID=123; Secure; Domain=example.com   # требует Secure
Set-Cookie: __Host-ID=123; Secure; Path=/                 # требует Secure + Path=/ + без Domain
```

```bash
# установка сессионного куки при входе
POST /login HTTP/1.1
Content-Type: application/json

{"username": "alice", "password": "secret"}

HTTP/1.1 200 OK
Set-Cookie: session=abc123xyz; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=86400
Content-Type: application/json

{"status": "ok"}

# следующий запрос — браузер отправляет куки автоматически
GET /api/profile HTTP/1.1
Cookie: session=abc123xyz

HTTP/1.1 200 OK
Content-Type: application/json

{"name": "Alice"}
```

# Set-Login

Отправляется федеративным провайдером идентификации (IdP) в рамках [FedCM API](https://developer.mozilla.org/en-US/docs/Web/API/FedCM_API) для обновления статуса авторизации пользователя в браузере.

```bash
Set-Login: logged-in   # пользователь вошёл в систему
Set-Login: logged-out  # пользователь вышел из системы
```

```bash
# IdP сообщает что пользователь вошёл
POST /idp/login HTTP/1.1
Host: idp.example.com

HTTP/1.1 200 OK
Set-Login: logged-in

# IdP сообщает что пользователь вышел
POST /idp/logout HTTP/1.1
Host: idp.example.com

HTTP/1.1 200 OK
Set-Login: logged-out
```

# SourceMap

Указывает расположение source map для минифицированного JavaScript или CSS. Позволяет браузерному DevTools отображать оригинальный исходный код.

```bash
SourceMap: <url>
SourceMap: /path/to/file.js.map
SourceMap: https://example.com/static/app.js.map
```

```bash
GET /static/app.min.js HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Content-Type: text/javascript
SourceMap: /static/app.min.js.map

(минифицированный JS)
```

# Strict-Transport-Security

Сообщает браузеру, что обращаться к хосту следует **только по HTTPS**. Браузер запоминает это на `max-age` секунд и автоматически заменяет любые http:// запросы на https:// без реального редиректа.

```bash
Strict-Transport-Security: max-age=<seconds>
Strict-Transport-Security: max-age=<seconds>; includeSubDomains  # распространить на поддомены
Strict-Transport-Security: max-age=<seconds>; includeSubDomains; preload  # внести в браузерный preload-список
```

```bash
Strict-Transport-Security: max-age=63072000; includeSubDomains; preload
# 63072000 = 2 года — рекомендуемое значение для добавления в preload-список
```

```bash
# 1. первый запрос по HTTP — сервер перенаправляет
GET http://example.com/ HTTP/1.1
Host: example.com

HTTP/1.1 301 Moved Permanently
Location: https://example.com/

# 2. запрос по HTTPS — сервер устанавливает HSTS
GET /page HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Strict-Transport-Security: max-age=63072000; includeSubDomains
Content-Type: text/html

# 3. все последующие запросы браузер автоматически делает по HTTPS
# (не ожидая редиректа)
```

# Timing-Allow-Origin

Указывает какие источники могут видеть детальные временны́е метрики Resource Timing API. Без этого заголовка cross-origin запросы видят нули в полях `redirectStart`, `fetchStart`, `domainLookupStart` и т.д.

```bash
Timing-Allow-Origin: *
Timing-Allow-Origin: https://developer.example.com
Timing-Allow-Origin: https://a.example.com, https://b.example.com
```

```bash
GET /api/data.json HTTP/1.1
Origin: https://app.example.com

HTTP/1.1 200 OK
Timing-Allow-Origin: https://app.example.com
Content-Type: application/json

{"data": "..."}
```

```js
// в браузере на app.example.com
const [entry] = performance.getEntriesByName(
  "https://api.example.com/data.json",
);
console.log(entry.domainLookupStart); // реальное значение (не 0)
console.log(entry.connectStart); // реальное значение (не 0)
```

# Vary

Перечисляет заголовки запроса, которые влияют на содержимое ответа. Кеш хранит отдельные копии ответа для каждой уникальной комбинации значений этих заголовков. Без `Vary` кеш может отдать сжатый ответ клиенту, не поддерживающему сжатие.

```bash
Vary: *                          # ответ уникален для каждого запроса, кешировать нельзя
Vary: Accept-Encoding            # разные копии по алгоритму сжатия
Vary: Accept-Language            # разные копии по языку
Vary: Origin                     # разные копии по origin (при динамическом CORS)
Vary: Accept-Encoding, Accept-Language
```

```bash
# сервер отдаёт разные сжатые версии — кеш хранит их по Accept-Encoding
GET /page.html HTTP/1.1
Accept-Encoding: br

HTTP/1.1 200 OK
Content-Encoding: br
Vary: Accept-Encoding
Content-Type: text/html

(brotli-сжатое тело)

# другой клиент без поддержки br
GET /page.html HTTP/1.1
Accept-Encoding: gzip

HTTP/1.1 200 OK
Content-Encoding: gzip
Vary: Accept-Encoding
Content-Type: text/html

(gzip-сжатое тело)
```

```bash
# при динамическом CORS — Vary: Origin обязателен чтобы кеш не перепутал ответы
GET /api/data HTTP/1.1
Origin: https://app.example.com

HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://app.example.com
Vary: Origin
Content-Type: application/json
```

# WWW-Authenticate

Содержит описание методов HTTP-аутентификации, применимых к ресурсу. Отправляется в ответе **401 Unauthorized**. Клиент выбирает схему и повторяет запрос с заголовком [Authorization](./req-headers.md#authorization).

```bash
# WWW-Authenticate: <scheme> realm="<realm>"
# WWW-Authenticate: <scheme> <token68>
# схемы: Basic, Digest, Bearer, Negotiate, AWS4-HMAC-SHA256, HOBA
```

**Basic:**

```bash
GET /secure HTTP/1.1
Host: example.com

HTTP/1.1 401 Unauthorized
WWW-Authenticate: Basic realm="My Site", charset="UTF-8"

GET /secure HTTP/1.1
Host: example.com
Authorization: Basic dXNlcjpwYXNzd29yZA==  # base64(user:password)

HTTP/1.1 200 OK
```

**Bearer (OAuth 2.0):**

```bash
GET /api/profile HTTP/1.1
Host: api.example.com

HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="api", error="invalid_token", error_description="Token expired"

GET /api/profile HTTP/1.1
Authorization: Bearer eyJhbGciOiJSUzI1NiJ9...

HTTP/1.1 200 OK
```

**Digest:**

```bash
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Digest
    realm="http-auth@example.org",
    qop="auth, auth-int",
    algorithm=SHA-256,
    nonce="7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v",
    opaque="FQhe/qaU925kfnzjCev0ciny7QMkPqMAFRtzCUYo5tdS"

Authorization: Digest username="Mufasa",
    realm="http-auth@example.org",
    uri="/dir/index.html",
    algorithm=SHA-256,
    nonce="7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v",
    nc=00000001,
    cnonce="f2/wE4q74E6zIJEtWaHKaf5wv/H5QzzpXusqGemxURZJ",
    qop=auth,
    response="8ca523f5e9506fed4657c9700eebdbec",
    opaque="FQhe/qaU925kfnzjCev0ciny7QMkPqMAFRtzCUYo5tdS"
```

**HOBA:**

```bash
HTTP/1.1 401 Unauthorized
WWW-Authenticate: HOBA max-age="180", challenge="16:MTEyMzEyMzEyMw==1:028:https://www.example.com:8080:3:MTI48:NjgxNDdjOTctNDYxYi00MzEwLWJlOWItNGM3MDcyMzdhYjUz"

Authorization: 123.16:MTEyMzEyMzEyMw==1:028:https://www.example.com:8080:3:MTI48:NjgxNDdjOTctNDYxYi00MzEwLWJlOWItNGM3MDcyMzdhYjUz.1123123123.<signature-of-challenge>
```
