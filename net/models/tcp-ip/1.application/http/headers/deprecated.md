устаревшие заголовки, которые в будущем будут выведены из оборота

# Attribution-Reporting-Eligible

Заголовок запроса, который браузер добавляет автоматически, сигнализируя серверу что ответ на этот запрос может зарегистрировать источник атрибуции или триггер Attribution Reporting API (измерение конверсий без сторонних куки).

```bash
# браузер отправляет на img/script/fetch запрос
Attribution-Reporting-Eligible: navigation-source

# возможные значения
Attribution-Reporting-Eligible: event-source
Attribution-Reporting-Eligible: trigger
Attribution-Reporting-Eligible: event-source, trigger
Attribution-Reporting-Eligible: navigation-source
```

# Attribution-Reporting-Register-Source

Заголовок ответа, регистрирующий источник атрибуции (клик или показ). JSON-объект описывает что и как отслеживать.

```bash
# запрос с рекламного баннера
GET /ad-click HTTP/1.1
Attribution-Reporting-Eligible: event-source

# ответ сервера — регистрация источника
HTTP/1.1 200 OK
Attribution-Reporting-Register-Source: {
  "source_event_id": "412444888111012",
  "destination": "https://advertiser.example",
  "expiry": 604800,
  "priority": 100
}
```

Ключевые поля JSON:

- `source_event_id` — уникальный ID источника
- `destination` — сайт конверсии
- `expiry` — срок жизни источника в секундах (по умолчанию 30 дней)
- `priority` — приоритет при нескольких источниках

# Attribution-Reporting-Register-Trigger

Заголовок ответа, регистрирующий триггер атрибуции (конверсию). Браузер сопоставляет триггер с сохранёнными источниками и отправляет отчёт.

```bash
# запрос на странице рекламодателя после покупки
GET /conversion-pixel HTTP/1.1
Attribution-Reporting-Eligible: trigger

# ответ сервера — регистрация триггера
HTTP/1.1 200 OK
Attribution-Reporting-Register-Trigger: {
  "event_trigger_data": [
    {
      "trigger_data": "1",
      "priority": "100",
      "deduplication_key": "orderid-7777"
    }
  ]
}
```

# Connection (req-res)

Используется только в HTTP/1.x. Управляет тем, остаётся ли TCP-соединение открытым после завершения текущего запроса. Все hop-by-hop заголовки должны перечисляться в Connection (Keep-Alive, Transfer-Encoding, TE, Trailer, Upgrade, Proxy-Authorization, Proxy-Authenticate) — прокси обязан их удалить перед пересылкой.

```bash
# клиент просит сохранить соединение
GET /page HTTP/1.1
Host: example.com
Connection: keep-alive

# сервер подтверждает
HTTP/1.1 200 OK
Connection: keep-alive
Keep-Alive: timeout=5, max=100
Content-Type: text/html

(body)

# клиент закрывает соединение
GET /logout HTTP/1.1
Host: example.com
Connection: close

HTTP/1.1 200 OK
Connection: close
```

В HTTP/2 и HTTP/3 Connection запрещён — управление соединением реализовано на уровне протокола.

# Content-DPR (req)

Заголовок ответа (несмотря на имя, отправляется сервером в ответ на DPR-запрос), сообщающий клиенту какой DPR использовался при выборе изображения. Позволяет браузеру корректно вычислить intrinsic size картинки.

```bash
# сервер запрашивает DPR и Width у клиента
HTTP/1.1 200 OK
Accept-CH: DPR, Width

# клиент отправляет свои параметры
GET /image.jpg HTTP/1.1
DPR: 2.0
Width: 400

# сервер возвращает изображение 800px и сообщает использованный DPR
HTTP/1.1 200 OK
Content-DPR: 2.0
Content-Type: image/jpeg
Content-Length: 54321
```

Замена — [Sec-CH-DPR](./sec-headers.md#sec-ch-dpr)

# Device-Memory (req)

Client hint, сообщающий приблизительный объём RAM устройства в ГБ. Значения округлены до ближайшей степени двойки для защиты приватности.

```bash
# сервер запрашивает подсказку
HTTP/1.1 200 OK
Accept-CH: Device-Memory
Vary: Device-Memory

# клиент отправляет
GET /app.js HTTP/1.1
Device-Memory: 4
# возможные значения: 0.25, 0.5, 1, 2, 4, 8

# сервер возвращает облегчённую версию для слабых устройств
GET /app.js HTTP/1.1
Device-Memory: 0.5
```

Замена — [Sec-CH-Device-Memory](./sec-headers.md#sec-ch-device-memory)

# DNT (req)

"Do Not Track" — клиент выражает предпочтение не быть отслеживаемым. Сервер отвечает заголовком [Tk](#tk). Большинство браузеров убрали поддержку, так как сайты его игнорировали.

```bash
# клиент отказывается от трекинга
DNT: 1

# клиент согласен на трекинг
DNT: 0

# клиент не выразил предпочтение
DNT: null
```

```bash
# полный диалог
GET /page HTTP/1.1
Host: example.com
DNT: 1

HTTP/1.1 200 OK
Tk: N
```

Замена — [Sec-GPC](./sec-headers.md#sec-gpc-ff)

# DPR (req)

Client hint, сообщающий device pixel ratio клиента — отношение физических пикселей к CSS-пикселям. Используется сервером для выбора изображения подходящего разрешения.

```bash
# сервер запрашивает DPR
HTTP/1.1 200 OK
Accept-CH: DPR

# клиент отправляет
GET /image.jpg HTTP/1.1
DPR: 2.0
# типичные значения: 1.0 (обычный), 2.0 (retina), 3.0 (high-dpi mobile)
```

Замена — [Sec-CH-DPR](./sec-headers.md#sec-ch-dpr)

# Expect-CT (res)

Заголовок ответа, требовавший от браузеров проверки сертификата через Certificate Transparency — публичные журналы, куда центры сертификации обязаны вносить все выданные сертификаты. Работал только в Chrome.

```bash
# только мониторинг нарушений
Expect-CT: max-age=86400, report-uri="https://example.com/ct-report"

# принудительное применение
Expect-CT: max-age=86400, enforce

# применение + отчёты
Expect-CT: max-age=86400, enforce, report-uri="https://example.com/ct-report"
```

Параметры:

- `max-age` — сколько секунд браузер должен соблюдать политику
- `enforce` — отклонять соединения при нарушении (без флага — только репортинг)
- `report-uri` — куда отправлять JSON-отчёт о нарушении

Замена — требование CT встроено в корневые программы браузеров (Chrome, Firefox) без необходимости заголовка.

# Keep-Alive

Заголовок HTTP Keep-Alive позволяет отправителю указать параметры постоянного соединения — таймаут и максимальное количество запросов.

Сообщение с Keep-Alive должно также содержать заголовок [Connection](#connection-req-res): keep-alive.

В протоколах HTTP/2 и HTTP/3 запрещены поля заголовка, специфичные для конкретного соединения, такие как [Connection](#connection-req-res) и Keep-Alive.

```bash
Keep-Alive: timeout=5, max=200

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

- `timeout` — минимальное время в секундах, в течение которого соединение должно оставаться открытым
- `max` — максимальное число запросов, которые можно отправить по этому соединению

# Pragma (req-res)

Устаревший заголовок управления кешированием HTTP/1.0. Единственное практически значимое значение — `no-cache`, идентичное `Cache-Control: no-cache`.

```bash
Pragma: no-cache # == Cache-Control: no-cache
```

```bash
# запрос клиента, требующего свежий ответ (старый клиент HTTP/1.0)
GET /data HTTP/1.0
Pragma: no-cache

# современный эквивалент
GET /data HTTP/1.1
Cache-Control: no-cache
```

Замена — [Cache-Control](./req-res-headers.md#cache-control)

# Report-To (res)

Устаревший заголовок для настройки эндпоинтов отчётности (CSP, Network Error Logging и др.). Принимал JSON с описанием группы эндпоинтов.

```bash
# устаревший формат
Report-To: {
  "group": "csp-endpoint",
  "max_age": 10886400,
  "endpoints": [
    { "url": "https://example.com/reports" }
  ]
}
Content-Security-Policy: default-src 'self'; report-to csp-endpoint
```

Замена — [Reporting-Endpoints](./res-headers.md#reporting-endpoints)

```bash
# современный эквивалент
Reporting-Endpoints: csp-endpoint="https://example.com/reports"
Content-Security-Policy: default-src 'self'; report-to csp-endpoint
```

# Tk

Заголовок ответа на запрос с [DNT](#dnt-req). Сообщает клиенту статус применённой политики трекинга.

```bash
GET /page HTTP/1.1
DNT: 1

HTTP/1.1 200 OK
Tk: N
```

Возможные значения:

- `!` — под следствием (сайт заявляет, что соответствует требованиям)
- `?` — статус неизвестен
- `G` — gateway (несколько сторон, политика применяется ко всем)
- `N` — не отслеживает
- `T` — отслеживает
- `C` — отслеживает с согласия пользователя
- `P` — потенциально отслеживает (не подтверждено)
- `D` — игнорирует DNT
- `U` — политика обновлена

# Trailer (req-res)

Позволяет добавлять дополнительные поля в конец chunked-сообщения. Полезен для метаданных, которые становятся известны только после передачи тела (например, контрольная сумма). Значение заголовка — список имён полей, которые появятся в трейлере.

```bash
# сервер анонсирует, какие поля будут в трейлере
HTTP/1.1 200 OK
Transfer-Encoding: chunked
Trailer: Expires, Checksum

# тело с чанками
7\r\n
Mozilla\r\n
9\r\n
Developer\r\n
7\r\n
Network\r\n
0\r\n
# трейлер — поля после нулевого чанка
Expires: Wed, 21 Oct 2025 07:28:00 GMT
Checksum: sha256=abc123...
\r\n
```

Запрещённые поля в трейлере: `Transfer-Encoding`, `Content-Length`, `Trailer` сам по себе.

# Viewport-Width

Client hint, сообщающий ширину экрана клиента в CSS-пикселях. Позволяет серверу выбрать изображение нужного размера.

```bash
# сервер запрашивает подсказку
HTTP/1.1 200 OK
Accept-CH: Viewport-Width
Vary: Viewport-Width

# клиент отправляет ширину viewport
GET /hero-image.jpg HTTP/1.1
Viewport-Width: 1280

# узкий экран получает меньшее изображение
GET /hero-image.jpg HTTP/1.1
Viewport-Width: 375
```

Замена — [Sec-CH-Viewport-Width](./sec-headers.md#sec-ch-viewport-width)

# Warning

Содержит информацию о возможных проблемах со статусом сообщения, например об устаревшем кеше. В ответе может содержаться более одного заголовка Warning.

```bash
# Warning: <warn-code> <warn-agent> <warn-text> [<warn-date>]
Warning: 110 anderson/1.3.37 "Response is stale"

Date: Wed, 21 Oct 2015 07:28:00 GMT
Warning: 112 - "cache down" "Wed, 21 Oct 2015 07:28:00 GMT"
```

Коды предупреждений:

- `110` — Response is Stale (ответ устарел)
- `111` — Revalidation Failed (не удалось проверить свежесть)
- `112` — Disconnected Operation (кеш работает в автономном режиме)
- `113` — Heuristic Expiration (эвристический срок истечения > 24 часов)
- `199` — Miscellaneous Warning (произвольное предупреждение)
- `214` — Transformation Applied (прокси применил преобразование)
- `299` — Miscellaneous Persistent Warning

# Width (req)

Client hint, сообщающий желаемую ширину ресурса в физических пикселях — внутренний размер изображения, который будет отображён. Комбинируется с [DPR](#dpr-req).

```bash
# сервер запрашивает подсказку
HTTP/1.1 200 OK
Accept-CH: Width, DPR

# клиент сообщает желаемую ширину и DPR
GET /product.jpg HTTP/1.1
Width: 800
DPR: 2.0

# сервер возвращает изображение 800px + сообщает использованный DPR
HTTP/1.1 200 OK
Content-DPR: 2.0
Content-Type: image/jpeg
```

Замена — [Sec-CH-Width](./sec-headers.md#sec-ch-width)
