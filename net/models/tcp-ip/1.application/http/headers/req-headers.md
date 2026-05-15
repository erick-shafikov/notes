# заголовки запроса

Заголовки по типу запроса клиента:

# Accept-Language

Сообщает серверу предпочитаемый язык. Значения генерируются из `navigator.languages`. q-фактор (0–1) задаёт приоритет: без q = 1.0 (максимум). Сервер может вернуть 406 если ни один язык не поддерживается.

```bash
Accept-Language: fr-CH, fr;q=0.9, en;q=0.8, de;q=0.7, *;q=0.5
# или просто
Accept-Language: de
```

```bash
# клиент указывает предпочтения
GET /docs HTTP/1.1
Accept-Language: ru, en;q=0.8

# сервер отвечает на русском и подтверждает язык
HTTP/1.1 200 OK
Content-Language: ru
Content-Type: text/html; charset=utf-8
```

# Alt-Used

Сообщает серверу какой альтернативный сервис был использован для соединения. Браузер добавляет этот заголовок автоматически после переключения на alt-сервис из [Alt-Svc](./res-headers.md#alt-svc). Нужен для диагностики и маршрутизации.

```bash
# 1. сервер сообщает о доступном альтернативном сервисе (HTTP/3 на порту 443)
HTTP/1.1 200 OK
Alt-Svc: h3=":443"; ma=86400

# 2. браузер переключается на HTTP/3 и сообщает об этом
GET /page HTTP/3
Alt-Used: example.com:443
```

# Authorization

Предоставляет реквизиты для аутентификации на сервере. Обычный flow: сервер сначала отвечает 401 с [WWW-Authenticate](./res-headers.md#www-authenticate), клиент повторяет запрос с Authorization.

```bash
Authorization: <auth-scheme> <authorization-parameters>
# auth-scheme: Basic, Digest, Bearer, AWS4-HMAC-SHA256
```

```bash
# 1. клиент делает запрос без авторизации
GET /protected HTTP/1.1
Host: example.com

# 2. сервер требует аутентификацию
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Basic realm="My Site", charset="UTF-8"

# 3. клиент отправляет credentials
GET /protected HTTP/1.1
Authorization: Basic dXNlcjpwYXNzd29yZA==
# dXNlcjpwYXNzd29yZA== = base64("user:password")

HTTP/1.1 200 OK
```

```bash
# Digest authentication — более безопасный вариант: пароль не передаётся в открытом виде
Authorization: Digest
    username="user",
    realm="http-auth@example.org",
    uri="/dir/index.html",
    algorithm=MD5,
    nonce="7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v",
    nc=00000001,
    cnonce="f2/wE4q74E6zIJEtWaHKaf5wv/H5QzzpXusqGemxURZJ",
    qop=auth,
    response="8ca523f5e9506fed4657c9700eebdbec",
    opaque="FQhe/qaU925kfnzjCev0ciny7QMkPqMAFRtzCUYo5tdS"

# Bearer — токен (JWT, OAuth2)
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

# Available-Dictionary

Передаёт хеш словаря (SHA-256 base64), а не строковый ID. Строковый ID словаря передаётся отдельным заголовком [Dictionary-ID](./experemental.md#dictionary-id-req-res--ff--sf). Полный диалог — в секции [Available-Dictionary](./experemental.md#available-dictionary).

```bash
GET /js/app.v2.js HTTP/1.1
Accept-Encoding: gzip, br, zstd, dcb, dcz
Available-Dictionary: :pZGm1Av0IEBKARczz7exkNYsZb8LzaMrV7J32a2fFG4=:
Dictionary-ID: "dict-v1"
```

# Cookie

Содержит хранимые cookie, ранее установленные сервером через [Set-Cookie](./res-headers.md#set-cookie) или через `document.cookie` в JS. Все cookie домена отправляются в одном заголовке.

```bash
Cookie: PHPSESSID=298zf09hf012fh2; csrftoken=u32t4o3tb3gg43; _gat=1
```

```bash
# 1. сервер устанавливает cookie
HTTP/1.1 200 OK
Set-Cookie: session=abc123; HttpOnly; Secure; SameSite=Lax
Set-Cookie: theme=dark

# 2. браузер автоматически отправляет cookie на следующий запрос
GET /dashboard HTTP/1.1
Cookie: session=abc123; theme=dark

HTTP/1.1 200 OK
```

# Expect

Указывает ожидание, которое сервер должен выполнить перед тем как клиент отправит тело. Единственное значение — `100-continue`. Используется для больших тел чтобы не тратить трафик если сервер всё равно откажет.

```bash
# 1. клиент сообщает что собирается загрузить большой файл и ждёт разрешения
PUT /upload/video HTTP/1.1
Host: origin.example.com
Content-Type: video/h264
Content-Length: 1234567890987
Expect: 100-continue

# 2. сервер разрешает — клиент отправляет тело
HTTP/1.1 100 Continue

(тело: 1.2ТБ видео)

# 3. сервер подтверждает загрузку
HTTP/1.1 201 Created

# или сервер сразу отказывает — тело не отправляется вовсе
HTTP/1.1 417 Expectation Failed
```

# Forwarded

Содержит информацию о цепочке прокси через которые прошёл запрос. Используется для дебага, статистики, определения реального IP клиента. Добавляется обратным прокси.

```bash
Forwarded: by=<identifier>;for=<identifier>;host=<host>;proto=<http|https>

Forwarded: for="_mdn"

# case insensitive
Forwarded: For="[2001:db8:cafe::17]:4711"

# separated by semicolon
Forwarded: for=192.0.2.60;proto=http;by=203.0.113.43

# цепочка через несколько прокси — через запятую
Forwarded: for=192.0.2.43, for=198.51.100.17
```

- `for` — клиент или предыдущий прокси
- `by` — текущий прокси (кто добавил заголовок)
- `host` — Host из исходного запроса
- `proto` — исходный протокол
- identifier: `_hidden`, `_secret` (обфусцированные — обязательно с `_`), IPv4, `[IPv6]`, `"unknown"`

Заменяет устаревшие [X-Forwarded-\*](./x-headers.md#x-forwarded-for-req):

```bash
X-Forwarded-For: 192.0.2.172
Forwarded: for=192.0.2.172

X-Forwarded-For: 192.0.2.43, 2001:db8:cafe::17
Forwarded: for=192.0.2.43, for="[2001:db8:cafe::17]"
```

# From

Содержит email-адрес оператора автоматизированного агента (бота, краулера). Позволяет владельцам сайтов связаться с оператором при избыточных запросах.

```bash
From: webmaster@example.org

GET /sitemap.xml HTTP/1.1
Host: example.com
User-Agent: MyBot/1.0
From: bot-admin@mycompany.com
```

# Host

Устанавливает хост и номер порта сервера куда отправляется запрос. Обязательный заголовок в HTTP/1.1. Если порта нет — по умолчанию 443 (https) и 80 (http). При отсутствии сервер ответит 400 Bad Request.

```bash
Host: example.com
Host: example.com:8080
Host: 192.0.2.1:3000

GET /page HTTP/1.1
Host: www.example.com
```

# If-Match

Условный запрос. Выполняется только если [ETag](./res-headers.md#etag) ресурса совпадает с одним из переданных значений, иначе 412. Сравнение — byte-by-byte (без `W/` префикса, то есть строгое).

Варианты использования:

- для PUT/PATCH — проверить что ресурс не изменился с момента последнего чтения (optimistic concurrency)
- для GET + [Range](./req-headers.md#range) — гарантировать что частичный запрос относится к тому же ресурсу

```bash
If-Match: "abc123"
If-Match: "abc123", "def456"
If-Match: *  # любой существующий ресурс
```

```bash
# 1. получаем ресурс с ETag
GET /articles/1 HTTP/1.1

HTTP/1.1 200 OK
ETag: "abc123"
Content-Type: application/json

{"title": "Hello", "body": "..."}

# 2. редактируем и сохраняем с проверкой ETag
PUT /articles/1 HTTP/1.1
If-Match: "abc123"
Content-Type: application/json

{"title": "Hello World", "body": "..."}

# если кто-то успел изменить ресурс раньше — ETag сменился
HTTP/1.1 412 Precondition Failed

# если ETag совпадает — сохраняем
HTTP/1.1 200 OK
ETag: "xyz789"
```

# If-Modified-Since

Условный запрос. Сервер возвращает 200 если ресурс изменился после указанной даты, иначе 304 Not Modified. Применяется только для GET и HEAD. Игнорируется если присутствует [If-None-Match](#if-none-match).

```bash
If-Modified-Since: <day-name>, <day> <month> <year> <hour>:<minute>:<second> GMT
```

```bash
# 1. первый запрос — получаем дату изменения
GET /styles.css HTTP/1.1

HTTP/1.1 200 OK
Last-Modified: Tue, 10 Oct 2023 10:00:00 GMT
Cache-Control: no-cache

(тело файла)

# 2. повторный запрос — кеш свежий, тело не нужно
GET /styles.css HTTP/1.1
If-Modified-Since: Tue, 10 Oct 2023 10:00:00 GMT

HTTP/1.1 304 Not Modified
Last-Modified: Tue, 10 Oct 2023 10:00:00 GMT

# 3. если файл изменился — сервер возвращает новое содержимое
HTTP/1.1 200 OK
Last-Modified: Mon, 20 Nov 2023 15:30:00 GMT

(новое тело файла)
```

# If-None-Match

Условный запрос. Сервер возвращает 200 только если ETag ресурса **не совпадает** ни с одним из переданных значений. Используется для обновления кешированной сущности. Предпочтительнее `If-Modified-Since` — ETag точнее.

```bash
If-None-Match: "abc123"
If-None-Match: "abc123", "def456"
If-None-Match: *  # вернуть 412 если ресурс уже существует (для безопасного создания)
```

```bash
# 1. получаем ресурс с ETag
GET /api/user/1 HTTP/1.1

HTTP/1.1 200 OK
ETag: "v3"
Content-Type: application/json

{"name": "Alice"}

# 2. повторный запрос — если ETag совпадает, ресурс не изменился
GET /api/user/1 HTTP/1.1
If-None-Match: "v3"

HTTP/1.1 304 Not Modified

# 3. если ETag изменился — сервер возвращает новые данные
HTTP/1.1 200 OK
ETag: "v4"

{"name": "Alice Smith"}
```

# If-Range

Условный запрос для возобновляемой загрузки. Если условие (ETag или дата) выполнено — сервер возвращает запрошенный диапазон (206). Если условие не выполнено (ресурс изменился) — возвращает весь ресурс (200). Используется вместе с [Range](#range).

```bash
If-Range: "abc123"                              # ETag
If-Range: Sat, 29 Oct 2023 19:43:31 GMT        # дата
```

```bash
# 1. начали скачивать, получили первые 500 байт
GET /file.zip HTTP/1.1

HTTP/1.1 200 OK
ETag: "abc123"
Content-Length: 10000

(первые 500 байт получены, соединение прервалось)

# 2. возобновляем скачивание — если файл не изменился, продолжаем с 500
GET /file.zip HTTP/1.1
Range: bytes=500-
If-Range: "abc123"

# файл не изменился — отдаём остаток
HTTP/1.1 206 Partial Content
Content-Range: bytes 500-9999/10000

(байты 500-9999)

# файл изменился — отдаём весь файл заново
HTTP/1.1 200 OK
ETag: "xyz789"

(весь файл с новым ETag)
```

# If-Unmodified-Since

Условный запрос. Выполняется только если ресурс **не был изменён** после указанной даты, иначе 412. Применяется для безопасного редактирования без ETag — гарантирует что не затрём чужие изменения.

```bash
If-Unmodified-Since: <day-name>, <day> <month> <year> <hour>:<minute>:<second> GMT
```

```bash
# читаем документ и запоминаем дату
GET /docs/spec.txt HTTP/1.1

HTTP/1.1 200 OK
Last-Modified: Mon, 01 Jan 2024 12:00:00 GMT

# правим и сохраняем — только если никто не трогал с тех пор
PUT /docs/spec.txt HTTP/1.1
If-Unmodified-Since: Mon, 01 Jan 2024 12:00:00 GMT
Content-Type: text/plain

(обновлённое содержимое)

# кто-то успел изменить раньше нас
HTTP/1.1 412 Precondition Failed

# никто не трогал — сохраняем
HTTP/1.1 200 OK
```

# Origin

Определяет источник (scheme + hostname + port), инициировавший запрос. Похож на [Referer](#referer), но не раскрывает путь. Браузер добавляет автоматически к cross-origin и CORS запросам.

```bash
Origin: https://example.com
Origin: https://example.com:8080
Origin: null  # sandboxed iframe, data: URI, etc.
```

```bash
GET /api/data HTTP/1.1
Origin: https://app.example.com

HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://app.example.com
```

Может быть `null`:

- схема не http/https/ftp/ws/wss/gopher (например, `blob:`, `file:`, `data:`)
- sandboxed iframe без `allow-same-origin`
- документ из `document.createDocument()`
- редирект с потерей origin
- Referrer-Policy установлена в `no-referrer`

# Prefer

Определяет предпочтительное поведение сервера при обработке запроса. Сервер не обязан выполнять предпочтения, но если выполнил — сообщает об этом заголовком [Preference-Applied](./res-headers.md#preference-applied).

```bash
Prefer: respond-async           # обработать асинхронно, вернуть 202
Prefer: return=minimal          # минимальный ответ (без тела)
Prefer: return=representation   # вернуть полное представление ресурса
Prefer: wait=10                 # ждать не более 10 секунд до асинхронного перехода
Prefer: handling=lenient        # игнорировать неизвестные предпочтения
Prefer: handling=strict         # вернуть ошибку при неизвестных предпочтениях
```

```bash
# минимальный ответ после создания — не возвращать тело ресурса
POST /resource HTTP/1.1
Host: example.com
Content-Type: application/json
Prefer: return=minimal

{"id":123, "name": "abc"}

HTTP/1.1 201 Created
Preference-Applied: return=minimal
Location: /resource/123

# асинхронная обработка долгой операции
POST /reports/generate HTTP/1.1
Prefer: respond-async, wait=5

HTTP/1.1 202 Accepted
Location: /reports/status/456
Preference-Applied: respond-async
```

# Proxy-Authorization

Предоставляет учётные данные для аутентификации на прокси-сервере. Отправляется в ответ на 407 Proxy Authentication Required с [Proxy-Authenticate](./res-headers.md#proxy-authenticate).

```bash
Proxy-Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
Proxy-Authorization: Bearer <token>
```

```bash
# 1. запрос через прокси без авторизации
GET https://example.com/ HTTP/1.1
Host: example.com

# 2. прокси требует аутентификацию
HTTP/1.1 407 Proxy Authentication Required
Proxy-Authenticate: Basic realm="Corporate Proxy"

# 3. клиент повторяет с учётными данными
GET https://example.com/ HTTP/1.1
Host: example.com
Proxy-Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l

HTTP/1.1 200 OK
```

# Range

Запрашивает часть ресурса. Сервер возвращает 206 Partial Content с заголовком [Content-Range](./representation-headers.md#content-range). При неверном диапазоне — 416. Если сервер не поддерживает range-запросы — возвращает 200 со всем ресурсом.

```bash
Range: bytes=0-499        # первые 500 байт
Range: bytes=900-         # с байта 900 до конца
Range: bytes=-100         # последние 100 байт
Range: bytes=200-999, 2000-2499, 9500-  # несколько диапазонов
```

```bash
# запрос первого мегабайта файла
GET /video.mp4 HTTP/1.1
Range: bytes=0-1048575

HTTP/1.1 206 Partial Content
Content-Range: bytes 0-1048575/52428800
Content-Length: 1048576
Content-Type: video/mp4

(первый мегабайт)

# неверный диапазон
GET /file.zip HTTP/1.1
Range: bytes=999999-1000000

HTTP/1.1 416 Range Not Satisfiable
Content-Range: bytes */5000
```

# Referer

Содержит URL страницы, с которой был инициирован запрос. Используется для аналитики, логирования и защиты от CSRF. Управляется политикой [Referrer-Policy](./res-headers.md#referrer-policy).

```bash
# переход по ссылке с главной страницы
GET /products/42 HTTP/1.1
Referer: https://example.com/

# запрос ресурса со страницы
GET /images/logo.png HTTP/1.1
Referer: https://example.com/about
```

# Service-Worker

Включается браузером в запросы на получение скрипта Service Worker. Позволяет серверу и администраторам отличать запросы sw-скрипта от обычных и вести мониторинг.

```bash
GET /sw.js HTTP/1.1
Host: example.com
Service-Worker: script

HTTP/1.1 200 OK
Content-Type: text/javascript
Service-Worker-Allowed: /
```

# TE

Указывает кодировки передачи (Transfer Encoding), которые клиент готов принять в ответе, и является ли поддержка трейлеров (`trailers`). Аналог [Accept-Encoding](./req-res-headers.md#accept-encoding) но для Transport-уровня, а не Content-уровня.

```bash
TE: trailers          # клиент поддерживает Trailer-поля в chunked-ответе
TE: gzip
TE: deflate
TE: gzip, trailers    # и сжатие, и трейлеры
```

```bash
GET /stream HTTP/1.1
TE: trailers, gzip;q=0.8

HTTP/1.1 200 OK
Transfer-Encoding: chunked, gzip
Trailer: Checksum

7\r\n
Mozilla\r\n
0\r\n
Checksum: sha256=abc123
\r\n
```

# Upgrade-Insecure-Requests

Клиент сигнализирует что предпочитает HTTPS. Сервер может ответить редиректом 301 на HTTPS или включить заголовок CSP с директивой `upgrade-insecure-requests`.

```bash
Upgrade-Insecure-Requests: 1
```

```bash
# клиент открывает http-страницу, но готов к https
GET / HTTP/1.1
Host: example.com
Upgrade-Insecure-Requests: 1

# сервер перенаправляет на HTTPS
HTTP/1.1 301 Moved Permanently
Location: https://example.com/
Vary: Upgrade-Insecure-Requests

# или CSP-директива автоматически апгрейдит вложенные ресурсы
HTTP/1.1 200 OK
Content-Security-Policy: upgrade-insecure-requests
```

# User-Agent

Строка идентификации клиента: приложение, движок, ОС, версия. Используется сервером для адаптации контента или аналитики. Формат: `<product>/<version> <comment>`.

```bash
# Firefox
User-Agent: Mozilla/5.0 (platform; rv:gecko-version) Gecko/gecko-trail Firefox/firefox-version
# Chrome
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36
# Opera
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 OPR/124.0.0.0 (Edition developer)
# Edge
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0
# Safari
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15
```
