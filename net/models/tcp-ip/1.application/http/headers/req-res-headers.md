# заголовки запроса и ответа

# Accept

клиент сообщает какие типы понимает. Какие типы контента MIME может понять, сервер возвращая [Content-Type](./res-headers.md#content-type) сервер сообщает какой тип отправил:

```bash
Accept: text/html, application/xhtml+xml, application/xml;q=0.9, _/_;q=0.8
```

# Accept-Encoding

сохранить одну копию, сжатую с помощью gzip, и другую — с помощью brotli. В паре Content-Encoding могут сервер может вернуть 406 Not Acceptable. Варианты по алгоритмам, могут использоваться с ;q=:

- - gzip - LZ77
- - compress - LZW
- - deflate - zlib + deflate
- - br - Brotli
- - zstd - Zstandard
- - dcb - Dictionary-Compressed Brotli
- - dcz - Dictionary-Compressed Zstandard
- - identity

В случае отсутствия 415 Unsupported Media Type

# Accept-Ranges

поддерживаются ли range-запросы, для частей ресурса

```bash
Accept-Ranges: bytes
```

# Cache-Control

определяет как будет работать кеш в браузере и промежуточных узлах - прокси и CDN. Основные понятия:

- cache
- Shared cache
- Private cache
- Store response
- Reuse response
- Revalidate response
- Fresh response
- Stale response
- Age

Значения делятся по следующим категориям

- Варианты управления кешированием:
- - public - может быть закеширован в любом месте
- - private - кеш для одного пользователя
- - no-cache - обязательный запрос на сервер при использования кешированных данных
- - only-if-cached - использование только закешированных данных
- Время жизни:
- - max-age=<seconds> - относительно от времени времени запроса
- - s-maxage=<seconds> - только для разделяемых кешей
- - min-fresh=<seconds> - запрос который актуален некоторое время
- - immutable - тело запроса не меняется
- Управление ре-валидацией и загрузкой:
- - must-revalidate - должен проверять
- - proxy-revalidate - только для прокси
- Другие:
- - no-store - не должно быть кеша
- - no-transform - не должны быть применимы преобразования Заголовки Content-Encoding, Content-Range, Content-Type не должны изменяться прокси

Значения по типу запроса:

- значения для запроса: max-age=<seconds>, max-stale[=seconds], min-fresh=seconds, no-cache, no-store, no-transform, only-if-cached
- значения для ответа: must-revalidate, no-cache, no-store, no-transform, public, private, proxy-revalidate, max-age=seconds, s-maxage=seconds

Инструкции (нестабильные):

- immutable
- stale-while-revalidate=<seconds>
- stale-if-error=<seconds>

соотношение запроса и ответа (req - res):

- max-age - max-age
- max-stale - X
- min-fresh - X
- X - s-maxage
- no-cache - no-cache
- no-store - no-store
- no-transform - no-transform
- only-if-cached - X
- X - must-revalidate
- X - proxy-revalidate
- X - must-understand
- X - private
- X - public
- X - immutable
- X - stale-while-revalidate
- stale-if-error - stale-if-error

```bash
# выключить кеш
Cache-Control: no-cache, no-store, must-revalidate
# статичный контент
Cache-Control: public, max-age=31536000
```

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

# Content-Disposition

В случае ответа, то как будет отображаться в браузере ответ
В случае запроса, то как интерпретировать каждую часть бинарных данных

Директивы:

- name - атрибут поля формы
- filename - Content-Disposition: attachment будет интерпретироваться "сохранить как"
- filename\* - filename но только с кодированием

```bash
# заголовки ответа
Content-Disposition: inline
Content-Disposition: attachment
Content-Disposition: attachment; filename="filename.jpg"
# заголовки составного запроса
Content-Disposition: form-data
Content-Disposition: form-data; name="fieldName"
Content-Disposition: form-data; name="fieldName"; filename="filename.jpg"
```

Пример ниже заставит браузер сохранить страницу под именем cool.html

```bash
# Ответ, вызывающий диалог "Сохранить как":
200 OK
Content-Type: text/html; charset=utf-8
Content-Disposition: attachment; filename="cool.html"
Content-Length: 22

<HTML>Save me!</HTML>
```

# Date

дата и время когда возникло сообщение

```bash
HTTP/1.1 200
Content-Type: text/html
Date: Tue, 29 Oct 2024 16:56:32 GMT

<html lang="en-US" …
```

# Keep-Alive

Заголовок HTTP Keep-Alive в запросе и ответе позволяет отправителю указать, как может использоваться соединение, например, по таймауту и ​​максимальному количеству запросов.

Keep-Alive сообщение с Keep-Alive должно также содержать заголовок Connection: keep-alive.

В протоколах HTTP/2 и HTTP/3 запрещены поля заголовка, специфичные для конкретного соединения, такие как Connection и Keep-Alive.

```bash
Keep-Alive: <parameters>

HTTP/1.1 200 OK
Connection: Keep-Alive
Content-Encoding: gzip
Content-Type: text/html; charset=utf-8
Date: Thu, 11 Aug 2016 15:23:13 GMT
Keep-Alive: timeout=5, max=200
Last-Modified: Mon, 25 Jul 2016 04:32:39 GMT
Server: Apache

(body)
```

Значения:

- timeout
- max

# Link

Указывает ссылку на метаданные ресурса. Семантика как link html элемента. Позволяет пред-загрузить ресурсы с preconnect и preload.

```bash
Link: <uri-reference>; param1=value1; param2="value2"
```

```bash
Link: <https://example.com>; rel="preconnect"
Link: <https://example.com/%E8%8B%97%E6%9D%A1>; rel="preconnect"
# несколько
Link: <https://one.example.com>; rel="preconnect", <https://two.example.com>; rel="preconnect", <https://three.example.com>; rel="preconnect"
```

использование для пагинации

```bash
Link: <https://api.example.com/issues?page=2>; rel="prev", <https://api.example.com/issues?page=4>; rel="next", <https://api.example.com/issues?page=10>; rel="last", <https://api.example.com/issues?page=1>; rel="first"
```

контроль приоритета загрузки

```bash
Link: </style.css>; rel=preload; as=style; fetchpriority="high"
```

# Priority

определяет порядок в котором ответ должен приходить, в ответе - в каком порядке пришли

```bash
Priority: u=<priority> # u - это число от 0 (самый высокий) до 7 (самый низкий)
Priority: i # incrementally - поступательное, по чанкам
Priority: u=<priority>, i
```

# Repr-Digest

предоставляет хеш-представление ресурса. Обобщает Content-Language, Content-Type, Content-Encoding. Может быть разным в зависимости от Content-Encoding и Content-Range

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

# Transfer-Encoding

Используется для определения типа кодирования сообщений. Hop-by-hop заголовок. Не рекомендуется для http2 и выше

```bash
Transfer-Encoding: chunked
Transfer-Encoding: compress # LZW
Transfer-Encoding: deflate # zlib
Transfer-Encoding: gzip # LZ77
Transfer-Encoding: gzip, chunked
```

# Upgrade

Используется для изменения протокола

```bash
# запрос на смену протокола
GET /index.html HTTP/1.1
Host: www.example.com
Connection: upgrade # всегда должен идти вместе Upgrade
Upgrade: example/1, foo/2

# подтверждение смены
HTTP/1.1 101 Switching Protocols
Upgrade: foo/2
Connection: Upgrade

# дальше идет ответ уже по новому протоколу
```

# Via

добавляется прокси-серверами, как прямыми, так и обратными. Он используется для отслеживания пересылаемых сообщений, предотвращения зацикливания запросов и определения возможностей протокола отправителей в цепочке запрос/ответ.

```bash
# Via: [<protocol-name>/]<protocol-version> <host>[:<port>]
# Via: [<protocol-name>/]<protocol-version> <pseudonym>
Via: 1.1 vegur
Via: HTTP/1.1 GWA
Via: 1.0 fred, 1.1 p.example.net
```

# Want-Content-Digest

Сигнал о том, что получатель рассчитывает получать [Content-Digest](#content-digest) В заголовках

```bash
# Want-Content-Digest: <algorithm>=<preference>
# Want-Content-Digest: <algorithm>=<preference>, …, <algorithmN>=<preferenceN>
Want-Content-Digest: sha-512=9
Want-Content-Digest: md5=1, sha-512=2, sha-256=3
```

# Want-Repr-Digest

казывает на предпочтение получателя отправлять заголовок целостности [Repr-Digest](#repr-digest) в сообщениях, связанных с URI запроса и метаданными представления.

```bash
Want-Repr-Digest: <algorithm>=<preference>
Want-Repr-Digest: <algorithm>=<preference>, …, <algorithmN>=<preferenceN>
```
