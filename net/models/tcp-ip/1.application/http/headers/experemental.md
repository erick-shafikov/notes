экспериментальные заголовки с частичной поддержкой

# Available-Dictionary

Заголовок запроса, сообщающий серверу хеш (SHA-256) словаря, который браузер закешировал ранее. Сервер может использовать этот словарь для дельта-сжатия ответа по протоколу Compression Dictionary Transport — ответ содержит только разницу, а не весь файл.

```bash
# 1. сервер отдаёт ресурс и помечает его как словарь для будущих запросов
HTTP/1.1 200 OK
Content-Type: application/javascript
Use-As-Dictionary: match="/js/app.*.js", id="dict-v1"

(тело: app.v1.js — браузер кеширует как словарь и считает SHA-256)

# 2. браузер запрашивает следующую версию, сообщая что словарь есть
GET /js/app.v2.js HTTP/1.1
Accept-Encoding: gzip, br, zstd, dcb, dcz
Available-Dictionary: :pZGm1Av0IEBKARczz7exkNYsZb8LzaMrV7J32a2fFG4=:
Dictionary-ID: "dict-v1"

# 3. сервер сжимает дельту относительно словаря
HTTP/1.1 200 OK
Content-Encoding: dcb
Content-Type: application/javascript

(тело: только изменения относительно app.v1.js)
```

# Critical-CH (res)

Сообщает браузеру, какие Client Hint заголовки критически важны для формирования ответа. Если браузер не отправил такой CH в текущем запросе — он обязан сделать повторный запрос уже с этим заголовком. В отличие от `Accept-CH`, который просто _просит_ CH на следующих запросах, `Critical-CH` требует повтора _этого_ запроса.

```bash
# 1. браузер делает начальный запрос без CH
GET / HTTP/1.1
Host: example.com

# 2. сервер сообщает что Prefers-Reduced-Motion критичен для рендеринга
HTTP/1.1 200 OK
Content-Type: text/html
Accept-CH: Sec-CH-Prefers-Reduced-Motion
Vary: Sec-CH-Prefers-Reduced-Motion
Critical-CH: Sec-CH-Prefers-Reduced-Motion

# 3. браузер автоматически повторяет запрос с нужным CH
GET / HTTP/1.1
Host: example.com
Sec-CH-Prefers-Reduced-Motion: "reduce"

# 4. сервер отвечает версией без анимаций
HTTP/1.1 200 OK
Content-Type: text/html
Vary: Sec-CH-Prefers-Reduced-Motion

(тело: страница без анимаций)
```

# Dictionary-ID (req-res) (-ff, -sf)

Отправляет строковый идентификатор словаря (в отличие от [Available-Dictionary](#available-dictionary), который передаёт хеш). Работает в паре с [Use-As-Dictionary](#use-as-dictionary): сервер назначает ID при отдаче словаря, клиент включает его в запросы для соответствующих ресурсов.

```bash
# 1. сервер отдаёт словарь с ID
HTTP/1.1 200 OK
Use-As-Dictionary: match="/js/app.*.js", id="dictionary-12345"

# 2. клиент запрашивает совпадающий ресурс, передавая и хеш, и ID
GET /js/app.v2.js HTTP/1.1
Accept-Encoding: gzip, br, zstd, dcb, dcz
Available-Dictionary: :pZGm1Av0IEBKARczz7exkNYsZb8LzaMrV7J32a2fFG4=:
Dictionary-ID: "dictionary-12345"
```

# Downlink (req)

Network client hint, сообщающий приблизительную полосу пропускания соединения клиента в **Мбит/с**. Сервер может использовать это для выбора качества медиа или объёма передаваемых данных.

```bash
# 1. сервер запрашивает подсказку
HTTP/1.1 200 OK
Accept-CH: Downlink
Vary: Downlink

# 2. клиент сообщает полосу пропускания
GET /video.mp4 HTTP/1.1
Downlink: 1.7
# типичные значения: 0.4 (slow-2g), 1.5 (2g), 3.0 (3g), 10.0 (4g), 100.0 (wifi)

# 3. сервер отдаёт подходящее качество
HTTP/1.1 200 OK
Content-Type: video/mp4
Vary: Downlink

(тело: видео 480p вместо 1080p)
```

# Early-Data (req)

Устанавливается прокси-посредником, чтобы сообщить серверу что запрос был передан в TLS 0-RTT Early Data (данные, отправленные до завершения полного TLS-хендшейка). Сервер может ответить [425 Too Early](../response-statuses.md), если операция небезопасна для повтора.

```bash
# посредник добавляет заголовок при пересылке 0-RTT запроса
GET /api/data HTTP/1.1
Host: example.com
Early-Data: 1

# сервер отклоняет небезопасный запрос (например, POST с деньгами)
HTTP/1.1 425 Too Early

# сервер принимает безопасный запрос (GET — идемпотентен)
HTTP/1.1 200 OK
```

# ECT (req)

"Effective Connection Type" — network client hint, сообщающий тип соединения клиента на основе измеренных RTT и Downlink. Используется сервером для адаптации ресурсов.

```bash
# сервер запрашивает подсказку
HTTP/1.1 200 OK
Accept-CH: ECT
Vary: ECT

# клиент сообщает тип соединения
GET /page HTTP/1.1
ECT: 4g
# значения: slow-2g (RTT > 2000мс), 2g (RTT > 1400мс), 3g (RTT > 270мс), 4g (RTT ≤ 270мс)
```

# Idempotency-Key (req)

Уникальный ключ для PATCH и POST запросов, позволяющий серверу отличить повторную попытку (retry) от нового запроса. Сервер сохраняет ключ и результат — при получении дубликата возвращает тот же ответ без повторного выполнения.

```bash
# первый запрос с уникальным UUID
POST /payments HTTP/1.1
Host: api.example.com
Content-Type: application/json
Idempotency-Key: "7da7a728-f910-11e6-942a-68f728c1ba70"

{"amount": 1000, "currency": "USD", "to": "alice"}

# успешный ответ — сервер сохраняет ключ
HTTP/1.1 201 Created
{"payment_id": "pay_123", "status": "completed"}

# повторный запрос с тем же ключом (например, из-за сетевой ошибки)
POST /payments HTTP/1.1
Idempotency-Key: "7da7a728-f910-11e6-942a-68f728c1ba70"

{"amount": 1000, "currency": "USD", "to": "alice"}

# сервер возвращает тот же результат без повторного списания
HTTP/1.1 201 Created
{"payment_id": "pay_123", "status": "completed"}

# дубликат с другим телом — конфликт
HTTP/1.1 409 Conflict
{"error": "Idempotency key reused with different request body"}
```

# NEL (res)

"Network Error Logging" — заголовок ответа, настраивающий браузер на отслеживание и репортинг сетевых ошибок (DNS-ошибки, TCP-сбои, TLS-проблемы, HTTP-ошибки). Отчёты отправляются на эндпоинт из [Reporting-Endpoints](./res-headers.md#reporting-endpoints).

```bash
# сервер настраивает NEL-политику
HTTP/1.1 200 OK
Reporting-Endpoints: network-errors="https://example.com/nel-reports"
NEL: {"report_to": "network-errors", "max_age": 2592000, "include_subdomains": true, "failure_fraction": 0.1, "success_fraction": 0.01}
```

Поля JSON:

- `report_to` — имя группы из Reporting-Endpoints
- `max_age` — сколько секунд действует политика (0 — удалить)
- `include_subdomains` — применять к поддоменам
- `failure_fraction` — доля ошибок для отправки (0.0–1.0)
- `success_fraction` — доля успехов для отправки (по умолчанию 0)

```bash
# пример отчёта который браузер отправит на эндпоинт
POST /nel-reports HTTP/1.1
Content-Type: application/reports+json

[{
  "type": "network-error",
  "url": "https://example.com/resource.js",
  "body": {
    "referrer": "https://example.com/",
    "sampling_fraction": 0.1,
    "server_ip": "203.0.113.1",
    "protocol": "http/1.1",
    "method": "GET",
    "status_code": 0,
    "elapsed_time": 143,
    "phase": "connection",
    "type": "tcp.refused"
  }
}]
```

# No-Vary-Search (res)

Указывает, какие query-параметры URL не должны учитываться при поиске закешированного ответа. Позволяет кешу считать URL с разными параметрами одинаковыми.

```bash
# порядок параметров не важен: ?a=1&b=2 == ?b=2&a=1
No-Vary-Search: key-order

# все параметры игнорируются: /page?any=value == /page
No-Vary-Search: params

# только указанные параметры игнорируются: /page?utm_source=x == /page
No-Vary-Search: params=("utm_source" "utm_medium" "utm_campaign")

# все параметры игнорируются, кроме указанных: "page" всё ещё влияет на кеш
No-Vary-Search: params, except=("page")

# комбинация: порядок не важен + utm_* игнорируются
No-Vary-Search: key-order, params=("utm_source" "utm_medium")
```

# Observe-Browsing-Topics (res)

Сообщает браузеру, что посещение этой страницы должно быть засчитано при формировании интересов пользователя в Topics API. Браузер использует эти данные для показа релевантной рекламы без передачи данных третьим сторонам.

```bash
# сервер сообщает: засчитать этот визит для Topics API
HTTP/1.1 200 OK
Observe-Browsing-Topics: ?1

# отключить наблюдение
Observe-Browsing-Topics: ?0
```

Значение `?1` / `?0` — булев тип в Structured Field Values (RFC 8941).

# Permissions-Policy (res)

Позволяет серверу разрешить или запретить браузерные API — как для текущего документа, так и для вложенных iframe.

Синтаксис allowlist:

- `feature=()` — запрещено везде (пустой список)
- `feature=*` — разрешено везде
- `feature=(self)` — только текущий origin
- `feature=(self "https://partner.com")` — текущий origin + указанный

```bash
# геолокация только для своего origin и партнёра, камера для всех, PiP отключён
Permissions-Policy: geolocation=(self "https://maps.example.com"), camera=*, picture-in-picture=()
```

```bash
# блокировка сторонних скриптов от использования микрофона
Permissions-Policy: microphone=(self)
```

```html
<!-- iframe с ограниченными правами -->
<iframe
  src="https://widget.example.com"
  allow="geolocation 'self' https://widget.example.com; camera 'none'"
></iframe>
```

разновидности:

- accelerometer
- ambient-light-sensor
- aria-notify
- attribution-reporting
- autoplay
- bluetooth
- browsing-topics
- camera
- captured-surface-control
- ch-ua-high-entropy-values
- compute-pressure
- cross-origin-isolated
- deferred-fetch
- deferred-fetch-minimal
- display-capture
- encrypted-media
- fullscreen
- gamepad
- geolocation
- gyroscope
- hid
- identity-credentials-get
- idle-detection
- language-detector
- local-fonts
- magnetometer
- microphone
- midi
- on-device-speech-recognition
- otp-credentials
- payment
- picture-in-picture
- private-state-token-issuance
- private-state-token-redemption
- publickey-credentials-create
- publickey-credentials-get
- screen-wake-lock
- serial
- speaker-selection
- storage-access
- summarizer
- translator
- usb
- web-share
- window-management
- xr-spatial-tracking

# Permissions-Policy-Report-Only (res)

Работает как [Permissions-Policy](#permissions-policy-res), но не блокирует запрещённые функции, а только отправляет отчёты о нарушениях. Используется для тестирования политики перед включением.

```bash
HTTP/1.1 200 OK
Reporting-Endpoints: policy-endpoint="https://example.com/policy-reports"
Permissions-Policy-Report-Only: geolocation=(), camera=(self)
```

# RTT (req)

"Round-Trip Time" — network client hint, сообщающий приблизительное время кругового пути на уровне приложения в миллисекундах. Позволяет серверу адаптировать количество или качество ресурсов под задержку сети.

```bash
# 1. сервер запрашивает подсказку
HTTP/1.1 200 OK
Accept-CH: RTT
Vary: RTT

# 2. клиент сообщает RTT
GET /page HTTP/1.1
RTT: 125
# типичные значения: 2400 (slow-2g), 900 (2g), 300 (3g), 50 (4g/wifi)

# 3. сервер отдаёт облегчённую версию при высоком RTT
HTTP/1.1 200 OK
Vary: RTT
```

# Save-Data (req)

Network client hint, сообщающий что клиент предпочитает минимальный трафик (например, мобильный интернет с лимитом). Сервер может отдавать сжатые изображения, убирать шрифты, отключать автовоспроизведение.

```bash
# request
GET /image.jpg HTTP/1.1
Host: example.com
Save-Data: on

# response
HTTP/1.1 200 OK
Content-Length: 102832
Vary: Accept-Encoding, Save-Data
Cache-Control: public, max-age=31536000
Content-Type: image/jpeg

[…]
```

# [Sec-заголовки](./sec-headers.md)

# Speculation-Rules (res) (-ff, -sf)

Содержит один или несколько URL на JSON-файлы с правилами спекулятивной навигации — браузер заранее prefetch или prerender ресурсы, ускоряя переход. Альтернатива тегу `<script type="speculationrules">`.

```bash
# сервер указывает URL с правилами
HTTP/1.1 200 OK
Speculation-Rules: "/speculation-rules.json"
```

```json
// /speculation-rules.json
{
  "prefetch": [
    {
      "urls": ["/next-page", "/popular-article"],
      "eagerness": "moderate"
    }
  ],
  "prerender": [
    {
      "where": { "href_matches": "/product/*" },
      "eagerness": "conservative"
    }
  ]
}
```

Параметр `eagerness`: `immediate`, `eager`, `moderate`, `conservative` — определяет насколько агрессивно браузер инициирует спекуляцию.

# Supports-Loading-Mode (res) (-ff, -sf)

Позволяет документу явно разрешить загрузку в контекстах, которые по умолчанию заблокированы — fenced frame или кросс-origin prerender.

Значения:

- `fenced-frame` — документ разрешает загружать себя внутри fenced frame
- `credentialed-prerender` — документ разрешает prerender с другого origin того же сайта (с учётными данными)

```bash
# разрешить загрузку внутри fenced frame
HTTP/1.1 200 OK
Supports-Loading-Mode: fenced-frame

# разрешить credentialed prerender с cross-origin (same-site)
HTTP/1.1 200 OK
Supports-Loading-Mode: credentialed-prerender
```

# Use-As-Dictionary

Заголовок ответа, указывающий браузеру сохранить текущий ресурс как словарь для Compression Dictionary Transport. При последующих запросах на совпадающие URL браузер отправит [Available-Dictionary](#available-dictionary) с хешем словаря.

```bash
# сервер помечает ресурс как словарь
HTTP/1.1 200 OK
Use-As-Dictionary: match="/js/app.*.js"

# с ID для Dictionary-ID заголовка
Use-As-Dictionary: match="/js/app.*.js", id="dict-v1"

# ограничить по типу назначения (script, style, document...)
Use-As-Dictionary: match="/js/app.*.js", match-dest=("script")

# тип содержимого словаря
Use-As-Dictionary: match="/js/app.*.js", type="raw"
```

Полный диалог — в секции [Available-Dictionary](#available-dictionary).
