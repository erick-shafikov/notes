# заголовки запроса

Заголовки по типу запроса клиента:

# Accept

клиент сообщает какие типы понимает. какие типы контента MIME может понять, сервер возвращая Content-Type сервер сообщает какой тип отправил:

```bash
Accept: text/html, application/xhtml+xml, application/xml;q=0.9, _/_;q=0.8
```

# Accept-Language

какой язык предпочитает клиент, это те же значения что и генерирует navigator.languages. Сервер может вернуть 406. Пример:

```bash
Accept-Language: fr-CH, fr;q=0.9, en;q=0.8, de;q=0.7, *;q=0.5
# или
Accept-Language: de
```

# Alt-Used

сообщает какой альтернативный сервис был использован используется вместе с Alt-Svc

##Authorization

предоставляет реквизиты для аутентификации на сервере

```bash
Authorization: <auth-scheme> <authorization-parameters>

# auth-scheme -  Basic, Digest, Negotiate,  AWS4-HMAC-SHA256.

# Basic authentication
Authorization: Basic <credentials>

# Digest authentication
Authorization: Digest username=<username>,
    realm="<realm>",
    uri="<url>",
    algorithm=<algorithm>,
    nonce="<nonce>",
    nc=<nc>,
    cnonce="<cnonce>",
    qop=<qop>,
    response="<response>", # hex
    opaque="<opaque>"
```

# Available-Dictionary

Это ID словаря, а не хеш

# Cookie

содержит хранимые cookie, которые были установлены Set-Cookie или js

```bash
Cookie: PHPSESSID=298zf09hf012fh2; csrftoken=u32t4o3tb3gg43; _gat=1
```

# Expect

Указывает ожидание, которое должен выполнить сервер. Единственно значение

```bash
Expect: 100-continue

# пример
PUT /somewhere/fun HTTP/1.1
Host: origin.example.com
Content-Type: video/h264
Content-Length: 1234567890987 # собираемся с клиента загрузить большой файл
Expect: 100-continue
```

Может быть 417

# Forwarded

Содержит информацию, которая может быть добавлена обратным прокси сервером. используется для дебага, статистики

```bash
Forwarded: by=<identifier>;for=<identifier>;host=<host>;proto=<http|https>

Forwarded: for="_mdn"

# case insensitive
Forwarded: For="[2001:db8:cafe::17]:4711"

# separated by semicolon
Forwarded: for=192.0.2.60;proto=http;by=203.0.113.43

# Values from multiple proxy servers can be appended using a comma
Forwarded: for=192.0.2.43, for=198.51.100.17
```

- identifier: "hidden", "secret", IPv4, IPv4v6,"unknown"
- host - Host заголовок
- proto - "http", "https"

Forwarded может заменить X-Forwarded-For <!-- вставить ссылку на X-Forwarded-For -->

```bash
X-Forwarded-For: 192.0.2.172
Forwarded: for=192.0.2.172

X-Forwarded-For: 192.0.2.43, 2001:db8:cafe::17
Forwarded: for=192.0.2.43, for="[2001:db8:cafe::17]"
```

# From

содержится интернет-адрес электронной почты администратора, управляющего автоматизированным пользовательским агентом. Если используется робот для запросов и количество запросов слишком большое, заголовок помогает связаться и решить проблему.

# Host

Устанавливает хост и номер порта сервера куда отправляется запрос, если порта нет то это будет 443 (https) и 80 (http). Если его нет, то сервер ответит 404

```bash
Host: <host>:<port>
```

# If-Match

Условный запрос. Позволяет сделать запрос только в том случае если [ETag](./res-headers.md#etag) совпадает со значением If-Match, в противном случае 412. Сравнение - byte-by-byte если без W\ префикса. Варианты использования:

- для get и head запросов в комбинации с Range <!-- Добавить ссылку--> гарантируют что будет запрос с одного ресурса
- для put запроса что бы узнать не изменен ди уже ресурс

```bash
If-Match: <etag_value>
If-Match: <etag_value>, <etag_value>, …
```

# If-Modified-Since

Условный запрос. Сервер на запрос с таким заголовком возвращает 200 если, если ресурс изменился после даты в If-Modified-Since header. Если не изменился то 304 с Last-Modified с предыдущей датой изменения. Применяется только для get и head запросов. Игнорируется если используется [If-None-Match](!!!TODOlink to if-none-match). Сигнатура - [даты](./res-headers.md#expires)

# If-None-Match

Условный запрос. Сервер возвращает запрошенный ресурс в методах GET и HEAD со статусом 200 только в том случае, если у него нет [ETag](./res-headers.md#etag), соответствующего значениям в заголовке If-None-Match. Нужен для обновления закешированной сущности с etag

# If-Range

Условный запрос. Если условие выполнено, отправляется запрос диапазона, и сервер возвращает ответ 206 Partial Content, содержащий часть (или части) ресурса в теле ответа. Использование - скачивание ресурса. Сигнатура - дата

# If-Unmodified-Since

Условный запрос. Сервер отправит запрошенный ресурс (или примет его в случае POST-запроса или другого небезопасного метода) только в том случае, если ресурс на сервере не был изменен после даты, указанной в заголовке запроса. Сигнатура - дата

# Origin

определяет источник scheme, hostname, port который инициировал запрос.Заголовок Origin похож на заголовок Referer, но не раскрывает путь и может быть пустым. Предоставляет источник запроса, когда он нужен. Используется:

- в cross origin запросах
- в same-origin запросах

Может быть null:

- если это не http, https, ftp, ws, wss, or gopher (blob, file, data)
- cross origin если это img, video, audio
- документ из createDocument()
- переправления на одном origin
- Документы, обслуживаемые с использованием директивы песочницы Content-Security-Policy, значение которой не включает allow-same-origin.
- iframe с атрибутом sandbox, значение которого не включает allow-same-origin. Ответы, являющиеся сетевыми ошибками.
- Для режимов запросов, не использующих CORS (например, для отправки простых форм), параметр Referrer-Policy должен быть установлен в значение no-referrer.

# Prefer

определяет поведение сервера при запросе с клиента

```bash
Prefer: respond-async, return=minimal, return=representation, wait=<seconds>, handling=lenient, handling=strict
```

пример для минимального ответа

```bash
# запрос
POST /resource HTTP/1.1
Host: example.com
Content-Type: application/json
Prefer: return=minimal

{"id":123, "name": "abc"}

# ответ

HTTP/1.1 201 Created
Location: /resource?id=123
```

# Proxy-Authorization

Предоставляет данные авторизации если сервер ответил 407 Proxy Authentication Required с [Proxy-Authenticate](./res-headers.md#proxy-authenticate)

```bash
Proxy-Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l
```

# Referer

Содержит абсолютный или частичный адрес откуда идет запрос. Нужен для отметки ресурсов, которые уже были посещены. Используется для кеширования, логирования

# Service-Worker

включается в запросы на получение ресурса скрипта сервис-воркера. Этот заголовок помогает администраторам регистрировать запросы скриптов сервис-воркера в целях мониторинга.

# TE

указываются кодировки передачи, которые пользовательский агент готов принять. Кодировки передачи используются для сжатия сообщений и разбиения данных на блоки во время передачи.

```bash
TE: compress
TE: deflate
TE: gzip
TE: trailers
```

# Upgrade-Insecure-Requests

отправляет сигнал что клиент предпочитает зашифрованный и аутентифицированный ответ

# User-Agent

представляет собой характерную строку, позволяющую серверам и сетевым узлам идентифицировать приложение, операционную систему, поставщика и/или версию запрашивающего пользовательского агента.

```bash
# User-Agent: <product> / <product-version> <comment>
# Firefox UA string
User-Agent: Mozilla/5.0 (platform; rv:gecko-version) Gecko/gecko-trail Firefox/firefox-version
# Chrome UA string
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36
# Opera UA string
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 OPR/124.0.0.0 (Edition developer)
# Microsoft Edge UA string
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0
# Safari UA string
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15
```
