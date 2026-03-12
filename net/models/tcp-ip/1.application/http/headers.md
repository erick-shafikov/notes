# заголовки

Делятся по контексту:

- те которые относятся к запросам и ответам
- заголовки запроса
- заголовки ответа
- заголовки репрезентативные
- заголовки тела запроса (payload)

Второй вариант группировки - как обрабатывают их прокси

Заголовки по типам: Connection, Keep-Alive, Proxy-Authenticate, Proxy-Authorization, TE, Trailer, Transfer-Encoding, Upgrade, сквозные, хоп-хоп заголовки

Некоторые заголовки поддерживают степень поддержки того или иного значения:

```bash
Accept-Encoding: br;q=1.0, gzip;q=0.8, \;q=0.1
```

# заголовки запроса

Заголовки по типу запроса клиента:

## Accept

клиент сообщает какие типы понимает. какие типы контента MIME может понять, сервер возвращая Content-Type сервер сообщает какой тип отправил:

```bash
Accept: text/html, application/xhtml+xml, application/xml;q=0.9, _/_;q=0.8
```

## Accept-Language

какой язык предпочитает клиент, это те же значения что и генерирует navigator.languages. Сервер может вернуть 406. Пример:

```bash
Accept-Language: fr-CH, fr;q=0.9, en;q=0.8, de;q=0.7, *;q=0.5
# или
Accept-Language: de
```

## Alt-Used

сообщает какой альтернативный сервис был использован используется вместе с Alt-Svc

## Authorization

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

## Available-Dictionary

Это ID словаря, а не хеш

<!------------------------------------------------------------>

# заголовки ответа

CH - client hint

Заголовки по типу ответ от сервера:

## Accept-CH

заголовок, который отправляет сервер, что бы получить CH заголовки от браузера, возможные значения: Sec-CH-UA-Model, Sec-CH-UA-Form-Factors, DPR, Viewport-Width, Width - сервер требует доп информацию, браузер на каждый запрос будет отправлять Sec-CH-UA-Model, Sec-CH-UA-Form-Factors. Поведение можно реализовать в html:

```html
<meta http-equiv="Accept-CH" content="Width, Downlink, Sec-CH-UA" />
```

Пример запроса:

```bash
Accept-CH: Sec-CH-Viewport-Width, Sec-CH-Width # отправил сервер
Vary: Sec-CH-Viewport-Width, Sec-CH-Width # какие vary заголовки ждет сервер
```

## Activate-Storage-Access

Это позволяет серверу активировать предоставленное разрешение на доступ к своим неразделенным файлам cookie в межсайтовом запросе.

```bash
Activate-Storage-Access: retry; allowed-origin="https://foo.bar"
Activate-Storage-Access: retry; allowed-origin=*
Activate-Storage-Access: load
```

## Activate-Storage-Access

заголовок предоставляет доступ к своим кукам для кросс запросов, сервер использует Sec-Fetch-Storage-Access заголовок

## Age

сколько объект запроса находился в кеше прокси

## Allow

Список дял доступных методов, должен возвращаться для 405 ошибки (Not Allowed)

## Alt-Svc

Приоритетный альтернативный ресурс

## Clear-Site-Data

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

<!---------------------------------------------------------------->

# заголовки запроса и ответа

## Accept-Encoding

сохранить одну копию, сжатую с помощью gzip, и другую — с помощью brotli. В паре Content-Encoding могут сервер может вернуть 406 Not Acceptable. Варианты по алгоритмам, могут использоваться с ;q=:

- - gzip - LZ77
- - compress - LZW
- - deflate - zlib + deflate
- - br - Brotli
- - zstd - Zstandard
- - dcb - Dictionary-Compressed Brotli
- - dcz - Dictionary-Compressed Zstandard
- - identity

## Accept-Ranges

поддерживаются ли range-запросы, для частей ресурса

## Cache-Control

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
- - no-transform - не должны быть применимы преобразования

Значения по типу запроса:

- значения для запроса: max-age=<seconds>, max-stale[=<seconds>], min-fresh=<seconds>, no-cache, no-store, no-transform, only-if-cached
- значения для ответа:must-revalidate, no-cache, no-store, no-transform, public, private, proxy-revalidate, max-age=<seconds>, s-maxage=<seconds>

Инструкции:

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

## Connection (dep)

использует только в http1, оставить ли соединение после запроса, значения keep-alive, close

## Content-Digest

алгоритм хеширования примененный к содержимому. Заголовок Want-Content-Digest запрашивает данные с хешированием, базируясь на Content-Encoding и Content-Range

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

<!---------------------------------------------------------------->

# репрезентативные и контентные заголовки

заголовки которые описывают как интерпретировать данные в сообщении

## Content-Encoding

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

## Content-Language

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

## Content-Length (res, req, cont)

размер в байтах. Передает информацию если идет стрим контента или генерация контента

<!---------------------------------------------------------------->

# заголовки OPTIONS (предварительного preflight-запроса)

## заголовки предварительного запроса

## Access-Control-Request-Headers

использует в предварительном запросе, для того что бы обозначить какие заголовки будет отправлять клиент

## Access-Control-Request-Method

использует в предварительном запросе, для того что бы обозначить какие методы будет отправлять клиент

## заголовки предварительного ответа

### Accept-Patch

Какие MIME типы может быть в теле PATCH запроса, должен быть в OPTIONS - запросе

### Accept-Post

Какие MIME типы может быть в теле POST запроса, должен быть в OPTIONS - запросе

### Access-Control-Allow-Credentials

Сообщает браузеру доступны ло cross-origin http запросы

```bash
Access-Control-Allow-Credentials: true
```

```js
// Allow credentials:
fetch(url, {
  credentials: "include",
});
```

### Access-Control-Allow-Headers

используется в предварительных запросах, какие http заголовки могут быть использованы. Пользовательский X-Custom-Header может быть использован:

```bash
Access-Control-Allow-Headers: X-Custom-Header
```

Используется в OPTIONS, пример запроса и ответа

```bash
# запрос
OPTIONS /resource/foo
Access-Control-Request-Method: GET
Access-Control-Request-Headers: content-type,x-requested-with
Origin: https://www.example.com
# ответ
HTTP/1.1 200 OK
Content-Length: 0
Connection: keep-alive
Access-Control-Allow-Origin: https://www.example.com
Access-Control-Allow-Methods: POST, GET, OPTIONS, DELETE
Access-Control-Allow-Headers: Content-Type, x-requested-with
Access-Control-Max-Age: 86400
```

### Access-Control-Allow-Methods

Позволяет определить http-методы доступные для запроса в предварительном запросе

### Access-Control-Allow-Origin

может ли быть доступен origin из источника

```bash
Access-Control-Allow-Origin: * # должен быть заголовок Vary
Access-Control-Allow-Origin: https://developer.mozilla.org

# при значении * должен быть заголовок Vary
Access-Control-Allow-Origin: https://developer.mozilla.org
Vary: Origin
```

### Access-Control-Expose-Headers

Какие заголовки ответа должны быть доступны скриптам, работающим в браузере, в ответ на междоменный запрос

### Access-Control-Max-Age

как долго живет результат предварительного запроса, значение в секундах

<!---------------------------------------------------------------->

# !!! предыдущее форматирование

- Заголовки устройства (отправляет браузер, далее б):
- - Sec-CH-UA-\* - подсказки клиента User Agent
- - Sec-CH-UA - версия браузера
- - Sec-CH-UA-Platform - платформа
- - Sec-CH-UA-Mobile: ?1 или ?2 - работает ли на мобильном устройстве
- - Sec-CH-UA-Model - модель
- - Sec-CH-UA-Form-Factors - форм факторы
- запрос на доп заголовки (от сервера, далее с)

- Кеширование:
- - Vary, значения:
- - - Cookie - каждый уникальный куки создает отдельную запись в кэше (что нередко приводит к катастрофическим последствиям)
- - - Vary: \* - означает «нельзя безопасно использовать этот ответ для других пользователей», что фактически отключает кэширование

- - Expires - пример: Expires: Wed, 29 Aug 2025 12:00:00 GMT
- - ETag - "abc123" - ресурс идентичен побайтово, W/"abc123" - семантически тот же
- - Last-Modified
- - Date - когда был сформирован ответ

- Критические
- - Sec-CH-Prefers-Reduced-Motion - может использоваться в Accept-CH
- Заголовки согласования контента, сервер сам определяет какой контент отдать:
- - Accept - mime типы
- - Accept-Encoding
- - Accept-Language
- - User-Agent
- Апгрейд протокола
- - Connection: upgrade - вернет 101 статус если изменит протокол или , также есть перечень заголовков для websocket протокола
- - Upgrade: example/1, foo/2

# устаревшие

- Attribution-Reporting-Eligible
- Attribution-Reporting-Register-Source
- Attribution-Reporting-Register-Trigger
- Content-DPR (RH) - подсказка для регулировки dpr изображения
