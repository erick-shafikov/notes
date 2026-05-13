# устаревшие

## Attribution-Reporting-Eligible

указывает на то, что соответствующий ответ подходит для регистрации источника атрибуции или триггера

## Attribution-Reporting-Register-Source

Это предоставляет информацию, которую браузер должен сохранять при взаимодействии пользователя с источником данных.

## Attribution-Reporting-Register-Trigger

Заголовок ответа HTTP Attribution-Reporting-Register-Trigger регистрирует функцию страницы в качестве триггера атрибуции.

## Connection (req-res)

использует только в http1, оставить ли соединение после запроса, значения keep-alive, close. Все hop-by-hop заголовки должны быть в Connection (Keep-Alive, Transfer-Encoding, TE, Connection, Trailer, Upgrade, Proxy-Authorization, Proxy-Authenticate). Connection можно устанавливать только заголовки, передаваемые на каждом узле соединения.

## Content-DPR (req)

подсказка для регулировки dpr изображения

## Device-Memory (req)

CH которая подсказывает объем памяти RAM девайса

## DNT (req)

отключает трекинг

## DPR (req)

о клиентском pixel ration

## Expect-CT (res)

заголовок, который управляет сертификатами (был только в хроме)

## Pragma (req-res)

управление кешированием HTTP/1.1

## Report-To (res)

указывает куда отправлять отчеты об нарушении CSP, заменен на директивы csp-endpoints или Reporting-Endpoints

## Tk

указывает на статус отслеживания, примененный к соответствующему запросу

## Trailer (req-res) (-)

позволяет отправителю добавлять дополнительные поля в конец фрагментированных сообщений для предоставления метаданных, которые могут динамически генерироваться во время отправки тела сообщения.

## Viewport-Width

Ширина в css-пикселях, замена - [Sec-CH-Viewport-Width](./sec-headers.md#sec-ch-viewport-width)

## Warning

Содержит информацию о возможных проблемах со статусом сообщения. В ответе может содержаться более одного заголовка Warning.

```bash
# Warning: <warn-code> <warn-agent> <warn-text> [<warn-date>]
# <warn-code: 110, 111, 112, 113, 199, 214,299
Warning: 110 anderson/1.3.37 "Response is stale"

Date: Wed, 21 Oct 2015 07:28:00 GMT
Warning: 112 - "cache down" "Wed, 21 Oct 2015 07:28:00 GMT"
```

# Width

указывает желаемую ширину ресурса в физических пикселях — внутренний размер изображения

```bash
# сервер должен инициировать
Accept-CH: Width
Width: 1920
```

# экспериментальные

## Critical-CH (res)

какие CH критичны

## Downlink (req)

CH полоса пропускания в МБ между клиентом и сервером

```bash
# первый запрос от сервера
Accept-CH: Downlink
# клиент отвечает
Downlink: 1.7
```

## Early-Data (req)

Заголовок запроса HTTP Early-Data устанавливается посредником, чтобы указать, что запрос был передан в формате ранних данных TLS, а также указать, что посредник понимает код состояния 425 Too Early.

## ECT (req)

клиентский тип соединения slow-2g, 2g, 3g, or 4g

## Idempotency-Key (req)

для patch и post запросов для работой с идемпотентными запросами, для которых нужен уникальный идентификатор. При нарушении - [400, 409, 422](../response-statuses.md). При 409 ответе (конфликт) в ответе желательно передать информацию

## NEL (res)

отвечает за логирование запроса

## No-Vary-Search (res)

указывает как параметры запроса влияют на сравнения для кеширование

## Observe-Browsing-Topics (res)

заголовок определяет интересы пользователя, использующиеся в iframe (Browsing-Topics API)

## Permissions-Policy (res)

позволяет принять или отклонить некоторые возможности браузера через iframe

```bash
<!-- разрешить использовать геолокации для всех источников-->
Permissions-Policy: geolocation=*
```

## Permissions-Policy-Report-Only (res)

для отчетов нарушений Permissions-Policy

## RTT (req)

это подсказка (network client hint) сетевого клиента, которая предоставляет приблизительное время кругового пути на уровне приложения в миллисекундах. Эта подсказка позволяет серверу выбирать, какая информация будет отправлена, в зависимости от скорости отклика/задержки сети. Например, он может выбрать отправку меньшего количества ресурсов.

```bash
# request
Accept-CH: RTT
# response
RTT: 125
```

## Save-Data (req)

это подсказка (network client hint) говорит о том что клиент предпочитает наименьшее потребление данных

```bash
# response
GET /image.jpg HTTP/1.1
Host: example.com
Save-Data: on

# request
HTTP/1.1 200 OK
Content-Length: 102832
Vary: Accept-Encoding, Save-Data
Cache-Control: public, max-age=31536000
Content-Type: image/jpeg

[…]
```

## [Sec-заголовки](./sec-headers.md)

## Speculation-Rules (res) (-ff, -sf)

содержит один или несколько URL-адресов, указывающих на текстовые ресурсы, содержащие определения правил спекуляции в формате JSON.

## Supports-Loading-Mode (res) (-ff, -sf)

Позволяет пользователю выбрать загрузку в новом, более рискованном контексте, в котором загрузка в противном случае не удалась бы.

```bash
# Ответ может быть загружен внутри изолированного фрейма. Без явного согласия на это, все навигации внутри изолированного фрейма завершатся неудачей.
Supports-Loading-Mode: fenced-frame
# Указывает на то, что источник данных выбирает загрузку документов посредством предварительной отрисовки на том же сайте, но с другого источника.
Supports-Loading-Mode: credentialed-prerender
```

# Use-As-Dictionary

В заголовке перечислены критерии соответствия, для которых может использоваться словарь Compression Dictionary Transport, для будущих запросов. Работает в паре с [Dictionary-ID](./res-headers.md#dictionary-id)

```bash
Use-As-Dictionary: match="<url-pattern>" # паттерн для пути который поддерживает вид словаря
Use-As-Dictionary: match-dest=("<destination1>" "<destination2>", …) # места вставки
Use-As-Dictionary: id="<string-identifier>" #  Dictionary-ID
Use-As-Dictionary: type="raw"


Content-Encoding: match="<url-pattern>", match-dest=("<destination1>")
```
