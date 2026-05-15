# репрезентативные и контентные заголовки

заголовки которые описывают как интерпретировать данные в сообщении

# Content-Digest

алгоритм хеширования примененный к содержимому. Заголовок [Want-Content-Digest](#want-content-digest) запрашивает данные с хешированием, базируясь на [Content-Encoding](./representation-headers.md#content-encoding) и [Content-Range](./representation-headers.md#content-range)

```bash
# digest-algorithm - sha-512 and sha-256. Небезопасные - md5, sha (SHA-1), unixsum, unixcksum, adler (ADLER32) and crc32c.
# digest-value - захешированное значение
Content-Digest: <digest-algorithm>=<digest-value>

# Multiple digest algorithms
Content-Digest: <digest-algorithm>=<digest-value>,<digest-algorithm>=<digest-value>, …
```

```bash
# запрос с клиента
GET /items/123 HTTP/1.1
Host: example.com
Want-Content-Digest: sha-256=10, sha=

# ответ с сервера
HTTP/1.1 200 OK
Content-Type: application/json
Content-Digest: sha-256=:RK/0qy18MlBSVnWgjwz6lZEWjP/lF5HF9bvEF8FabDg=:

# {"hello": "world"}
```

# Content-Encoding

Тип сжатия тела ответа. Клиент объявляет поддерживаемые алгоритмы через [Accept-Encoding](./req-res-headers.md#accept-encoding), сервер выбирает один и сообщает его в Content-Encoding. Клиент декодирует тело перед использованием.

```bash
Content-Encoding: gzip     # LZ77
Content-Encoding: compress # LZW
Content-Encoding: deflate  # zlib
Content-Encoding: identity # без сжатия
Content-Encoding: br       # Brotli
Content-Encoding: zstd     # Zstandard

# последовательное применение нескольких алгоритмов (раскодировать в обратном порядке)
Content-Encoding: deflate, gzip
```

```bash
# клиент сообщает что понимает
GET /page HTTP/1.1
Accept-Encoding: br, gzip;q=0.8, deflate;q=0.6

# сервер выбирает br и сжимает тело
HTTP/1.1 200 OK
Content-Type: text/html; charset=utf-8
Content-Encoding: br
Vary: Accept-Encoding

(brotli-сжатое тело)
```

# Content-Language

Сообщает к какой языковой аудитории предназначен контент. Клиент объявляет предпочтения через [Accept-Language](./req-headers.md#accept-language), сервер выбирает язык и отражает его в Content-Language.

```bash
Content-Language: de-DE
Content-Language: en-US
Content-Language: de-DE, en-CA  # контент на нескольких языках
```

```bash
# клиент указывает предпочтительный язык
GET /docs HTTP/1.1
Accept-Language: fr-CH, fr;q=0.9, en;q=0.8

# сервер возвращает французскую версию
HTTP/1.1 200 OK
Content-Language: fr-CH
Content-Type: text/html; charset=utf-8
```

Может быть указан в html-документе:

```html
<html lang="de"></html>
<!-- /!\ Это плохая практика -->
<meta http-equiv="content-language" content="de" />
```

# Content-Length (res, req)

Размер тела в байтах. Используется когда размер известен заранее. При стриминге или динамической генерации Content-Length опускается и вместо него используется [Transfer-Encoding: chunked](./req-res-headers.md#transfer-encoding).

```bash
# ответ со статичным файлом — размер известен
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 42

{"user": "alice", "role": "admin"}

# POST-запрос с известным телом
POST /upload HTTP/1.1
Content-Type: application/octet-stream
Content-Length: 1048576

(бинарное тело 1 МБ)
```

# Content-Location

Указывает прямой URL возвращённого представления при [согласовании контента](../сontent-negotiation.md). Отличается от `Location`: тот предполагает 3xx-редирект, а `Content-Location` — это информация о текущем ответе (нет редиректа).

```bash
# клиент запрашивает JSON
GET /documents/foo HTTP/1.1
Accept: application/json

# сервер сообщает конкретный URL этого представления
HTTP/1.1 200 OK
Content-Type: application/json
Content-Location: /documents/foo.json

{"title": "foo", "body": "..."}

# клиент запрашивает XML
GET /documents/foo HTTP/1.1
Accept: application/xml

HTTP/1.1 200 OK
Content-Type: application/xml
Content-Location: /documents/foo.xml

<document><title>foo</title></document>
```

```bash
# при создании ресурса — Content-Location указывает где его найти
POST /payments HTTP/1.1
Content-Type: application/json

{"amount": 500, "to": "alice"}

HTTP/1.1 201 Created
Content-Location: /payments/txn-9921
Content-Type: application/json

{"id": "txn-9921", "status": "completed"}
```

# Content-Range

Показывает положение передаваемой части ресурса относительно полного размера. Отправляется в ответах 206 Partial Content и 416 Range Not Satisfiable.

```bash
# клиент запрашивает первый мегабайт файла
GET /video.mp4 HTTP/1.1
Range: bytes=0-1048575

# сервер возвращает запрошенный диапазон
HTTP/1.1 206 Partial Content
Content-Type: video/mp4
Content-Length: 1048576
Content-Range: bytes 0-1048575/52428800

(бинарные данные)

# запрошенный диапазон выходит за пределы файла
HTTP/1.1 416 Range Not Satisfiable
Content-Range: bytes */52428800

# общий размер файла неизвестен (например, генерируется динамически)
Content-Range: bytes 0-1023/*
```

# Content-Type

определяет медиа тип ресурса, при ответе с сервера. при post и put запросах определяет тип отправляемого контента. Content-Encoding говорит о том как декодировать. Если не поддерживается тип - 415

```bash
# Content-Type: text/html; charset=utf-8
# Content-Type: multipart/form-data; boundary=ExampleBoundaryString
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
--ExampleBoundaryString--

# application/json
HTTP/1.1 201 Created
Content-Type: application/json

{
  "message": "New user created",
  "user": {
    "id": 123,
    "firstName": "Paul",
    "lastName": "Klee",
    "email": "p.klee@example.com"
  }
}
```

# Repr-Digest

предоставляет хеш-представление ресурса. Обобщает [Content-Language, Content-Type, Content-Encoding](./representation-headers.md). Может быть разным в зависимости от Content-Encoding и Content-Range

```bash
# Repr-Digest: <digest-algorithm>=<digest-value>
# Repr-Digest: <digest-algorithm>=<digest-value>,…,<digest-algorithmN>=<digest-valueN>

# request:
POST /bank_transfer HTTP/1.1
Host: example.com
Content-Encoding: zstd
Content-Digest: sha-512=:ABC…=:
Repr-Digest: sha-512=:DEF…=:

{
 "recipient": "Alex",
 "amount": 900000000
}
# response:
…
Repr-Digest: sha-256=:AEGPTgUMw5e96wxZuDtpfm23RBU3nFwtgY5fw4NYORo=:
Content-Digest: sha-256=:AEGPTgUMw5e96wxZuDtpfm23RBU3nFwtgY5fw4NYORo=:
…
Content-Type: text/yaml
Content-Encoding: br
Content-Length: 38054
Content-Range: bytes 0-38053/38054
…

[message body]
```

При успехе 201 Created. При несовпадении дайджеста сервер может вернуть 400 Bad Request (тело запроса изменено) или 406 Not Acceptable (алгоритм не поддерживается)

# Want-Content-Digest

Запрашивает у партнёра включить [Content-Digest](#content-digest) в ответные сообщения. Числовое значение (0–9) — предпочтение алгоритма: `0` означает «не использовать», `1–9` — уровень желательности (выше = предпочтительнее).

```bash
# Want-Content-Digest: <algorithm>=<preference>
# Want-Content-Digest: <algorithm>=<preference>, …, <algorithmN>=<preferenceN>

# запросить sha-256 с максимальным предпочтением
Want-Content-Digest: sha-256=9

# предпочесть sha-256, допустить sha-512, запретить md5
Want-Content-Digest: sha-256=9, sha-512=3, md5=0
```

```bash
# клиент просит включить дайджест в ответ
GET /items/123 HTTP/1.1
Want-Content-Digest: sha-256=9

# сервер включает Content-Digest
HTTP/1.1 200 OK
Content-Type: application/json
Content-Digest: sha-256=:RK/0qy18MlBSVnWgjwz6lZEWjP/lF5HF9bvEF8FabDg=:

{"id": 123, "name": "widget"}
```

# Want-Repr-Digest

Запрашивает у партнёра включить [Repr-Digest](#repr-digest) в ответные сообщения. Работает аналогично [Want-Content-Digest](#want-content-digest), но запрашивает дайджест всего представления (без учёта Content-Encoding и Content-Range), а не только передаваемого содержимого.

```bash
# Want-Repr-Digest: <algorithm>=<preference>
Want-Repr-Digest: sha-256=9
Want-Repr-Digest: sha-256=9, sha-512=3

# клиент запрашивает дайджест представления
GET /resource HTTP/1.1
Want-Repr-Digest: sha-256=9

# сервер отвечает дайджестом полного представления
HTTP/1.1 200 OK
Content-Type: application/json
Content-Encoding: gzip
Repr-Digest: sha-256=:d435Qo...=:

(gzip-сжатое тело — но дайджест считался по несжатому JSON)
```
