# заголовки запроса и ответа

# Accept

Клиент перечисляет MIME-типы, которые он готов принять. Сервер выбирает один из них и сообщает выбранный тип в [Content-Type](./representation-headers.md#content-type). Если ни один из указанных типов не поддерживается — 406 Not Acceptable.

Приоритет задаётся через `q=` (quality factor, 0–1). Чем выше значение — тем предпочтительнее тип.

```bash
# Accept: <MIME_type>/<MIME_subtype>
# Accept: <MIME_type>/*          — любой подтип
# Accept: */*                    — любой тип

Accept: text/html
Accept: text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8
```

```bash
# клиент запрашивает HTML, принимает XML с низким приоритетом, остальное — с ещё ниже
GET /page HTTP/1.1
Host: example.com
Accept: text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8

# сервер выбирает предпочтительный тип и сообщает его
HTTP/1.1 200 OK
Content-Type: text/html; charset=utf-8

<!DOCTYPE html>...
```

```bash
# если ни один тип не поддерживается
GET /resource HTTP/1.1
Accept: application/x-custom-type

HTTP/1.1 406 Not Acceptable
```

# Accept-Encoding

Клиент перечисляет поддерживаемые алгоритмы сжатия. Сервер выбирает один и сжимает тело, указывая использованный алгоритм в [Content-Encoding](./representation-headers.md#content-encoding). Если ни один алгоритм не поддерживается — 406 Not Acceptable.

Приоритет задаётся через `q=`, значение `identity` (без сжатия) всегда допустимо если не исключено явно.

```bash
# алгоритмы
# gzip      — LZ77
# compress  — LZW
# deflate   — zlib + deflate
# br        — Brotli
# zstd      — Zstandard
# dcb       — Dictionary-Compressed Brotli
# dcz       — Dictionary-Compressed Zstandard
# identity  — без сжатия

Accept-Encoding: gzip
Accept-Encoding: gzip, br;q=0.9, deflate;q=0.6
Accept-Encoding: *
```

```bash
# клиент сообщает поддерживаемые алгоритмы с приоритетами
GET /page HTTP/1.1
Host: example.com
Accept-Encoding: br, gzip;q=0.8, deflate;q=0.6

# сервер выбирает br и сжимает тело
HTTP/1.1 200 OK
Content-Type: text/html; charset=utf-8
Content-Encoding: br
Vary: Accept-Encoding

(brotli-сжатое тело)
```

```bash
# клиент не поддерживает ни один из доступных алгоритмов
GET /page HTTP/1.1
Accept-Encoding: compress

HTTP/1.1 406 Not Acceptable
```

# Accept-Ranges

Сервер сообщает, поддерживает ли он range-запросы (частичная загрузка ресурса). При `bytes` клиент может запросить отдельный диапазон байт через заголовок [Range](./req-headers.md#range). При `none` — range-запросы не поддерживаются.

```bash
Accept-Ranges: bytes  # поддерживаются байтовые диапазоны
Accept-Ranges: none   # range-запросы не поддерживаются
```

```bash
# клиент проверяет поддержку range-запросов
HEAD /video.mp4 HTTP/1.1
Host: example.com

# сервер сообщает о поддержке
HTTP/1.1 200 OK
Content-Length: 52428800
Accept-Ranges: bytes

# клиент запрашивает первый мегабайт
GET /video.mp4 HTTP/1.1
Host: example.com
Range: bytes=0-1048575

# сервер возвращает запрошенный диапазон
HTTP/1.1 206 Partial Content
Content-Type: video/mp4
Content-Length: 1048576
Content-Range: bytes 0-1048575/52428800
Accept-Ranges: bytes

(бинарные данные первого мегабайта)
```

# Cache-Control

Управляет кешированием в браузере и на промежуточных узлах (прокси, CDN). Применяется как в запросе (клиент задаёт требования к кешу), так и в ответе (сервер задаёт политику хранения).

**Основные понятия:**

- **Private cache** — кеш только для одного пользователя (браузер)
- **Shared cache** — кеш для нескольких пользователей (прокси, CDN)
- **Fresh response** — кешированный ответ, чей срок не истёк
- **Stale response** — ответ с истёкшим сроком, требует ревалидации
- **Revalidation** — проверка на сервере, актуален ли кешированный ответ

**Директивы управления кешированием:**

```bash
public              # можно кешировать где угодно, в том числе в shared cache
private             # только в private cache (браузере)
no-cache            # всегда запрашивать сервер для ревалидации перед использованием кеша
no-store            # не хранить ни в каком кеше вообще
only-if-cached      # использовать только закешированный ответ, без запроса к серверу
```

**Директивы времени жизни:**

```bash
max-age=<seconds>            # максимальный возраст ответа относительно времени запроса
s-maxage=<seconds>           # то же, но только для shared cache (перекрывает max-age)
min-fresh=<seconds>          # клиент принимает ответ только если он будет свежим ещё X секунд
stale-while-revalidate=<s>   # (нестабильная) отдавать устаревший кеш, пока идёт ревалидация
stale-if-error=<s>           # (нестабильная) использовать устаревший кеш при ошибке сервера
```

**Директивы ревалидации:**

```bash
must-revalidate     # при устаревании — обязательная ревалидация, нельзя отдать stale
proxy-revalidate    # то же, но только для прокси
must-understand     # кешировать только если понимает код статуса
immutable           # тело ответа не изменится в течение max-age (не загружать повторно)
```

**Прочее:**

```bash
no-transform        # промежуточные узлы не должны изменять Content-Encoding, Content-Range, Content-Type
max-stale[=<s>]     # клиент принимает устаревший ответ (не старше X секунд)
```

**Соответствие директив запрос → ответ:**

| Запрос         | Ответ                  |
| -------------- | ---------------------- |
| max-age        | max-age                |
| max-stale      | —                      |
| min-fresh      | —                      |
| —              | s-maxage               |
| no-cache       | no-cache               |
| no-store       | no-store               |
| no-transform   | no-transform           |
| only-if-cached | —                      |
| —              | must-revalidate        |
| —              | proxy-revalidate       |
| —              | must-understand        |
| —              | private                |
| —              | public                 |
| —              | immutable              |
| —              | stale-while-revalidate |
| stale-if-error | stale-if-error         |

```bash
# отключить кеш полностью
Cache-Control: no-cache, no-store, must-revalidate

# статичный контент с длинным сроком (fingerprinted URL)
Cache-Control: public, max-age=31536000, immutable

# HTML-документ: всегда проверять, но можно хранить
Cache-Control: no-cache

# приватные данные пользователя
Cache-Control: private, max-age=300
```

```bash
# 1. браузер запрашивает страницу
GET /index.html HTTP/1.1
Host: example.com

# 2. сервер отвечает с кеш-политикой на 1 час
HTTP/1.1 200 OK
Content-Type: text/html
Cache-Control: public, max-age=3600
ETag: "abc123"
Date: Mon, 12 May 2025 10:00:00 GMT

(html body)

# 3. через час кеш устарел, браузер ревалидирует
GET /index.html HTTP/1.1
Host: example.com
If-None-Match: "abc123"

# 4. сервер подтверждает: ничего не изменилось
HTTP/1.1 304 Not Modified
Cache-Control: public, max-age=3600
ETag: "abc123"
```

```bash
# клиент требует свежий ответ, игнорируя кеш
GET /api/data HTTP/1.1
Cache-Control: no-cache

HTTP/1.1 200 OK
Cache-Control: no-cache, no-store
Content-Type: application/json

{"time": "2025-05-12T10:00:00Z"}
```

# Content-Disposition

В **ответе** управляет тем, как браузер отображает содержимое: `inline` — встроить в страницу (по умолчанию), `attachment` — предложить сохранить файл.

В **multipart-запросах** (загрузка форм) описывает каждую часть тела: имя поля формы и имя файла.

**Директивы:**

```bash
inline                              # отображать в браузере (текст, изображения, PDF)
attachment                          # скачать как файл
attachment; filename="name.ext"     # скачать с именем файла
attachment; filename*=UTF-8''name.ext  # имя с non-ASCII символами (RFC 5987)

form-data; name="field"             # часть multipart: поле формы
form-data; name="file"; filename="upload.jpg"  # часть multipart: файл
```

```bash
# ответ с предложением скачать файл
GET /report.pdf HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Content-Type: application/pdf
Content-Disposition: attachment; filename="report-2025.pdf"
Content-Length: 204800

(бинарное содержимое PDF)
```

```bash
# multipart-запрос с полем формы и файлом
POST /upload HTTP/1.1
Host: example.com
Content-Type: multipart/form-data; boundary=ExampleBoundaryString
Content-Length: 1024

--ExampleBoundaryString
Content-Disposition: form-data; name="description"

Моё фото
--ExampleBoundaryString
Content-Disposition: form-data; name="photo"; filename="avatar.jpg"
Content-Type: image/jpeg

(бинарное содержимое фото)
--ExampleBoundaryString--
```

# Date

Дата и время формирования сообщения. Используется в обоих направлениях — сервер указывает когда ответ был создан, промежуточные узлы могут добавлять свои. Формат — HTTP-date по RFC 7231 (всегда GMT).

```bash
Date: <day-name>, <day> <month> <year> <hour>:<minute>:<second> GMT

Date: Tue, 29 Oct 2024 16:56:32 GMT
```

```bash
GET /page HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Date: Tue, 29 Oct 2024 16:56:32 GMT
Content-Type: text/html
Cache-Control: max-age=3600

(body)
```

# Link

Передаёт метаданные о связанных ресурсах — HTTP-аналог HTML-элемента `<link>`. Используется для управления предзагрузкой ресурсов (preload, preconnect), пагинацией API и указания на альтернативные представления.

```bash
Link: <uri-reference>; rel="<relation>"
Link: <uri-reference>; rel="<relation>"; param="value"
```

```bash
# пред-загрузка критических ресурсов
HTTP/1.1 200 OK
Link: </style.css>; rel=preload; as=style
Link: </font.woff2>; rel=preload; as=font; crossorigin
Link: <https://cdn.example.com>; rel=preconnect

# несколько ссылок через запятую
Link: <https://one.example.com>; rel="preconnect", <https://two.example.com>; rel="preconnect"
```

```bash
# пагинация REST API
GET /api/issues?page=3 HTTP/1.1
Host: api.example.com

HTTP/1.1 200 OK
Link: <https://api.example.com/issues?page=2>; rel="prev",
      <https://api.example.com/issues?page=4>; rel="next",
      <https://api.example.com/issues?page=1>; rel="first",
      <https://api.example.com/issues?page=10>; rel="last"
Content-Type: application/json

[...]
```

```bash
# Early Hints (103) для ускорения загрузки
HTTP/1.1 103 Early Hints
Link: </style.css>; rel=preload; as=style; fetchpriority=high
Link: </app.js>; rel=preload; as=script

HTTP/1.1 200 OK
Content-Type: text/html
Link: </style.css>; rel=preload; as=style
```

# Priority

Задаёт приоритет обработки запроса или ответа. В **запросе** клиент сообщает серверу насколько срочен ресурс. В **ответе** сервер подтверждает или корректирует приоритет. Используется в HTTP/2 и HTTP/3.

```bash
# u — urgency, 0 (самый высокий приоритет) до 7 (самый низкий)
# i — incremental: ресурс обрабатывается по чанкам по мере поступления
Priority: u=<0-7>
Priority: u=<0-7>, i
```

Значения по умолчанию для разных типов ресурсов:

```bash
Priority: u=0   # критические ресурсы (скрипты блокирующие рендер)
Priority: u=1   # высокий приоритет
Priority: u=3   # изображения (по умолчанию)
Priority: u=5   # фоновые задачи
Priority: u=7   # наименее важные
```

```bash
# клиент запрашивает главный CSS с высоким приоритетом
GET /critical.css HTTP/1.1
Host: example.com
Priority: u=0

# изображение с низким приоритетом
GET /hero-image.jpg HTTP/1.1
Host: example.com
Priority: u=6

HTTP/1.1 200 OK
Content-Type: image/jpeg
Priority: u=6
Content-Length: 204800

(изображение)
```

```bash
# потоковый ответ (JSON stream) с флагом incremental
GET /stream HTTP/1.1
Priority: u=3, i

HTTP/1.1 200 OK
Content-Type: application/json
Transfer-Encoding: chunked
Priority: u=3, i

7\r\n
{"id":1\r\n
...
```

# Transfer-Encoding

Определяет способ кодирования тела при передаче. Hop-by-hop заголовок — применяется только между двумя соседними узлами, не сохраняется при проксировании. Не рекомендуется в HTTP/2 и HTTP/3 (используется встроенное мультиплексирование).

```bash
Transfer-Encoding: chunked           # тело разбито на чанки неизвестного заранее размера
Transfer-Encoding: gzip              # тело сжато gzip (LZ77)
Transfer-Encoding: compress          # тело сжато LZW
Transfer-Encoding: deflate           # тело сжато zlib
Transfer-Encoding: gzip, chunked     # сначала сжато, потом разбито на чанки
```

**Формат chunked-тела:** каждый чанк начинается со строки с размером в hex, затем данные, в конце — нулевой чанк.

```bash
HTTP/1.1 200 OK
Content-Type: text/plain
Transfer-Encoding: chunked

7\r\n
Mozilla\r\n
9\r\n
Developer\r\n
7\r\n
Network\r\n
0\r\n
\r\n
```

```bash
# сервер стримит данные неизвестного размера
GET /live-log HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Content-Type: text/plain
Transfer-Encoding: chunked

1a\r\n
abcdefghijklmnopqrstuvwxyz\r\n
8\r\n
12345678\r\n
0\r\n
\r\n
```

```bash
# WebSocket upgrade — сервер переключает протокол
GET /chat HTTP/1.1
Host: example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==

HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Transfer-Encoding: chunked  # использовался до переключения
```

# Upgrade

Запрашивает переключение на другой протокол в рамках текущего соединения. Всегда сопровождается `Connection: Upgrade`. Сервер может принять (101 Switching Protocols) или отклонить (426 Upgrade Required с перечнем допустимых протоколов).

```bash
Upgrade: <protocol>[/<version>]
Upgrade: websocket
Upgrade: HTTP/2.0
Upgrade: example/1, foo/2  # список в порядке предпочтения
```

```bash
# клиент запрашивает переключение на WebSocket
GET /chat HTTP/1.1
Host: www.example.com
Connection: Upgrade
Upgrade: websocket
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

# сервер подтверждает переключение
HTTP/1.1 101 Switching Protocols
Connection: Upgrade
Upgrade: websocket
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

# далее идёт обмен по протоколу WebSocket
```

```bash
# сервер требует переключения на TLS
HTTP/1.1 426 Upgrade Required
Connection: Upgrade
Upgrade: TLS/1.3

(body)
```

# Via

Добавляется промежуточными узлами (прокси, шлюзами) как в прямых, так и в обратных запросах. Каждый узел добавляет свою запись в конец заголовка. Используется для отслеживания маршрута, предотвращения циклов и определения возможностей узлов.

```bash
# Via: [<protocol-name>/]<protocol-version> <host>[:<port>]
# Via: [<protocol-name>/]<protocol-version> <pseudonym>

Via: 1.1 vegur
Via: HTTP/1.1 GWA
Via: 1.0 fred, 1.1 p.example.net  # несколько узлов через запятую
```

```bash
# запрос проходит через два прокси
GET /resource HTTP/1.1
Host: origin.example.com

# proxy-a добавляет себя
GET /resource HTTP/1.1
Host: origin.example.com
Via: 1.1 proxy-a.example.com

# proxy-b добавляет себя следующим
GET /resource HTTP/1.1
Host: origin.example.com
Via: 1.1 proxy-a.example.com, 1.1 proxy-b.example.com

# origin отвечает, ответ идёт обратно через те же прокси
HTTP/1.1 200 OK
Content-Type: application/json
Via: 1.1 proxy-b.example.com, 1.1 proxy-a.example.com

{"data": "..."}
```
