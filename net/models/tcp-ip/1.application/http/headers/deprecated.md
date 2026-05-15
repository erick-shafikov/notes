устаревшие заголовки, которые в будущем будут выведены из оборота

# Attribution-Reporting-Eligible

указывает на то, что соответствующий ответ подходит для регистрации источника атрибуции или триггера

# Attribution-Reporting-Register-Source

Это предоставляет информацию, которую браузер должен сохранять при взаимодействии пользователя с источником данных.

# Attribution-Reporting-Register-Trigger

Заголовок ответа HTTP Attribution-Reporting-Register-Trigger регистрирует функцию страницы в качестве триггера атрибуции.

# Connection (req-res)

использует только в http1, оставить ли соединение после запроса, значения keep-alive, close. Все hop-by-hop заголовки должны быть в Connection (Keep-Alive, Transfer-Encoding, TE, Connection, Trailer, Upgrade, Proxy-Authorization, Proxy-Authenticate). Connection можно устанавливать только заголовки, передаваемые на каждом узле соединения.

# Content-DPR (req)

подсказка для регулировки dpr изображения

# Device-Memory (req)

CH которая подсказывает объем памяти RAM девайса

# DNT (req)

отключает трекинг

# DPR (req)

о клиентском pixel ration

# Expect-CT (res)

заголовок, который управляет сертификатами (был только в хроме)

# Keep-Alive

Заголовок HTTP Keep-Alive в запросе и ответе позволяет отправителю указать, как может использоваться соединение, например, по таймауту и ​​максимальному количеству запросов.

Keep-Alive сообщение с Keep-Alive должно также содержать заголовок Connection: keep-alive.

В протоколах HTTP/2 и HTTP/3 запрещены поля заголовка, специфичные для конкретного соединения, такие как [Connection](#connection-req-res) и Keep-Alive.

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

# Pragma (req-res)

управление кешированием HTTP/1.1

```bash
Pragma: no-cache # == Cache-Control: no-cache.
```

# Report-To (res)

указывает куда отправлять отчеты об нарушении CSP, заменен на директивы csp-endpoints или Reporting-Endpoints

# Tk

указывает на статус отслеживания, примененный к соответствующему запросу

# Trailer (req-res) (-)

позволяет отправителю добавлять дополнительные поля в конец фрагментированных сообщений для предоставления метаданных, которые могут динамически генерироваться во время отправки тела сообщения.

# Viewport-Width

Ширина в css-пикселях, замена - [Sec-CH-Viewport-Width](./sec-headers.md#sec-ch-viewport-width)

# Warning

Содержит информацию о возможных проблемах со статусом сообщения. В ответе может содержаться более одного заголовка Warning.

```bash
# Warning: <warn-code> <warn-agent> <warn-text> [<warn-date>]
# <warn-code: 110, 111, 112, 113, 199, 214,299
Warning: 110 anderson/1.3.37 "Response is stale"

Date: Wed, 21 Oct 2015 07:28:00 GMT
Warning: 112 - "cache down" "Wed, 21 Oct 2015 07:28:00 GMT"
```

# Width (req)

указывает желаемую ширину ресурса в физических пикселях — внутренний размер изображения

```bash
# сервер должен инициировать
Accept-CH: Width
Width: 1920
```
