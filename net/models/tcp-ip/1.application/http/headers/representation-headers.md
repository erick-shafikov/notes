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

тип сжатия body

```bash
Content-Encoding: gzip # LZ77
Content-Encoding: compress  # LZW
Content-Encoding: deflate  # zlib
Content-Encoding: identity # без сжатия
Content-Encoding: br # Brotli

# последовательность сжатий
Content-Encoding: gzip, identity
Content-Encoding: deflate, gzip
```

вместе с [Accept-Encoding](#accept-encoding) могут договариваться с сервером по поводу сжатия

# Content-Language

Сообщает о том к какой языковой группе относится контент

```bash
Content-Language: de-DE
Content-Language: en-US
Content-Language: de-DE, en-CA
```

Может быть указан в html Документе

```html
<html lang="de"></html>
<!-- /!\ Это плохая практика -->
<meta http-equiv="content-language" content="de" />
```

# Content-Length (res, req)

размер в байтах. Передает информацию если идет стрим контента или генерация контента

# Content-Location

альтернативное расположение возвращаемых данных, в отличие от Location указывает прямую ссылку при [согласовании контента](../сontent-negotiation.md). Location предполагает 3ХХ код ответа для редиректа

```bash
# Запрос
Accept: application/json, text/json
# Ответ
Content-Location: /documents/foo.json
# Запрос
Accept: application/xml, text/xml
# Ответ
Content-Location: /documents/foo.xml
# Запрос
Accept: text/plain, text/*
# Ответ
Content-Location: /documents/foo.txt
```

Как пример - при создании выплаты в Content-Location может быть помещен путь где находится страница с результатом выплаты

# Content-Range

показывает где находится содержимое тело ответа относительно ресурса

```bash
HTTP/2 206
content-type: image/jpeg
content-length: 1024
content-range: bytes 0-1023/146515
…

(binary content)

# или при неопределенном размере
Content-Range: bytes */67589
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
---------------------------1003363413119651595289485765

#  application/json
HTTP/1.1 201 Created
Content-Type: application/json

{
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
Content-Range: 0-38053/38054
…

[message body]
```

При успехе 201 Created, при ошибке 406 Not Acceptable

# Want-Content-Digest

Сигнал о том, что получатель рассчитывает получать [Content-Digest](#repr-digest) В заголовках

```bash
# Want-Content-Digest: <algorithm>=<preference>
# Want-Content-Digest: <algorithm>=<preference>, …, <algorithmN>=<preferenceN>
Want-Content-Digest: sha-512=9
Want-Content-Digest: md5=1, sha-512=2, sha-256=3
```

# Want-Repr-Digest

Заголовок запроса и ответа HTTP Want-Repr-Digest указывает на предпочтение получателя отправлять заголовок целостности Repr-Digest в сообщениях, связанных с URI запроса и метаданными представления.
