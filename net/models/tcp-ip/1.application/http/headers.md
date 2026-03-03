# заголовки

Делятся по контексту:

- те которые относятся к запросам и ответам
- заголовки запроса
- заголовки ответа
- заголовки сущности

Второй вариант группировки - как обрабатывают их прокси

Заголовки по типам: Connection, Keep-Alive, Proxy-Authenticate, Proxy-Authorization, TE, Trailer, Transfer-Encoding, Upgrade, сквозные, хоп-хоп заголовки

Некоторые заголовки поддерживают степень поддержки того или иного значения:

```bash
Accept-Encoding: br;q=1.0, gzip;q=0.8, \;q=0.1
```

# заголовки запроса

Заголовки по типу запроса клиента:

## Accept

какие типы контента MIME может понять, сервер возвращая Content-Type сообщает о наличие:

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
Accept-CH: Sec-CH-Viewport-Width, Sec-CH-Width - отправил сервер
Vary: Sec-CH-Viewport-Width, Sec-CH-Width - какие vary заголовки ждет сервер
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

<!---------------------------------------------------------------->

# заголовки запроса и ответа

## Accept-Encoding

сохранить одну копию, сжатую с помощью gzip, и другую — с помощью brotli. В паре Content-Encoding могут сгенерировать 415 ошибку. Варианты по алгоритмам, могут использоваться с ;q=:

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

<!---------------------------------------------------------------->

# заголовки сущности

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
Access-Control-Allow-Origin: * # все
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
- - - Accept-Language - хранить отдельные версии для en-US и de-DE
- - - Cookie - каждый уникальный куки создает отдельную запись в кэше (что нередко приводит к катастрофическим последствиям)
- - - Vary: \* - означает «нельзя безопасно использовать этот ответ для других пользователей», что фактически отключает кэширование

- - Cache-Control, значения:
- - - max-age - кол-во секунд сколько считать ответ актуальным
- - - s-maxage - для общих кешей
- - - immutable - никогда не изменится
- - - stale-while-revalidate - позволяет отдать устаревший, пока запрашивается актуальный
- - - stale-if-error - отдаст устаревший, если ошибка сервера
- - - Cache-Control, варианты для ответа:
- - - - public — ответ может храниться любым кэшем, включая общие
- - - - private — ответ может кэшироваться только браузером, общие кэши использовать его не могут
- - - - no-cache — хранить, но проверять актуальность перед выдачей
- - - - no-store — не хранить вообще
- - - - must-revalidate — после устаревания ответ нужно обязательно проверить перед использованием
- - - Cache-Control, варианты для запроса:
- - - - no-cache — принудительная проверка актуальности (при этом можно использовать уже сохраненные записи)
- - - - no-store — полностью игнорировать кэш
- - - - only-if-cached — вернуть ответ только из кэша, если он доступен; иначе — ошибка (полезно для работы офлайн)
- - - - max-age, min-fresh, max-stale — позволяют гибко управлять допустимым временем устаревания ответа
- - - proxy-revalidate — то же самое, но для общих кэшей

- - Expires - пример: Expires: Wed, 29 Aug 2025 12:00:00 GMT
- - ETag - "abc123" - ресурс идентичен побайтово, W/"abc123" - семантически тот же
- - Last-Modified
- - Date - когда был сформирован ответ
- - Age - у общих кешей

- Критические
- - Sec-CH-Prefers-Reduced-Motion - может использоваться в Accept-CH
- Заголовки согласования контента - не являются частью стандарта, сервер сам определяет какой контент отдать:
- - Accept - mime типы
- - Accept-Encoding
- - Accept-Language
- - User-Agent
- Апгрейд протокола
- - Connection: upgrade - вернет 101 статус если изменит протокол или , также есть перечень заголовков для websocket протокола
- - Upgrade: example/1, foo/2
