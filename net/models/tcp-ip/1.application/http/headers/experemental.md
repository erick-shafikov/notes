экспериментальные заголовки с частичной поддержкой

# Available-Dictionary

позволяет браузеру указать наиболее подходящий словарь, чтобы сервер мог использовать протокол Compression Dictionary Transport для запроса ресурса

# Critical-CH (res)

какие CH критичны

```bash
# req
GET / HTTP/1.1
Host: example.com

# res
HTTP/1.1 200 OK
Content-Type: text/html
Accept-CH: Sec-CH-Prefers-Reduced-Motion
Vary: Sec-CH-Prefers-Reduced-Motion
Critical-CH: Sec-CH-Prefers-Reduced-Motion
```

# Dictionary-ID (req-res) (-ff, -sf)

отправляет id [словаря](../compression-dictionary-transport.md) работает в паре с Use-As-Dictionary и с активным Available-Dictionary заголовками. Сервер отправляет [Use-As-Dictionary](#use-as-dictionary) а клиент отправляет хеш, который сервер проверяет [Available-Dictionary](#available-dictionary)

```bash
# ответ сервера при запросе ресурса
Use-As-Dictionary: match="/js/app.*.js", id="dictionary-12345"
# при запросе из браузера ресурса указывается Dictionary-ID: "dictionary-12345"
Accept-Encoding: gzip, br, zstd, dcb, dcz
Available-Dictionary: :pZGm1Av0IEBKARczz7exkNYsZb8LzaMrV7J32a2fFG4=:
Dictionary-ID: "dictionary-12345"
```

# Downlink (req)

CH полоса пропускания в МБ между клиентом и сервером

```bash
# первый запрос от сервера
Accept-CH: Downlink
# клиент отвечает
Downlink: 1.7
```

# Early-Data (req)

Заголовок запроса HTTP Early-Data устанавливается посредником, чтобы указать, что запрос был передан в формате ранних данных TLS, а также указать, что посредник понимает код состояния 425 Too Early.

# ECT (req)

клиентский тип соединения slow-2g, 2g, 3g, or 4g

# Idempotency-Key (req)

для patch и post запросов для работой с идемпотентными запросами, для которых нужен уникальный идентификатор. При нарушении - [400, 409, 422](../response-statuses.md). При 409 ответе (конфликт) в ответе желательно передать информацию

# NEL (res)

отвечает за логирование запроса

# No-Vary-Search (res)

указывает как параметры запроса влияют на сравнения для кеширование

```bash
No-Vary-Search: key-order
No-Vary-Search: params
No-Vary-Search: params=("param1" "param2")
No-Vary-Search: params, except=("param1" "param2")
No-Vary-Search: key-order, params, except=("param1")
```

# Observe-Browsing-Topics (res)

заголовок определяет интересы пользователя, использующиеся в iframe (Browsing-Topics API)

# Permissions-Policy (res)

позволяет принять или отклонить некоторые возможности браузера через iframe

```bash
# разрешить использовать геолокации для всех источников
Permissions-Policy: picture-in-picture=(), geolocation=(self https://example.com/), camera=*

Permissions-Policy: picture-in-picture=()
Permissions-Policy: geolocation=(self https://example.com/)
Permissions-Policy: camera=*
```

c iframe

```html
<iframe
  src="https://example.com"
  allow="geolocation 'self' https://a.example.com https://b.example.com"
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

для отчетов нарушений Permissions-Policy

# RTT (req)

это подсказка (network client hint) сетевого клиента, которая предоставляет приблизительное время кругового пути на уровне приложения в миллисекундах. Эта подсказка позволяет серверу выбирать, какая информация будет отправлена, в зависимости от скорости отклика/задержки сети. Например, он может выбрать отправку меньшего количества ресурсов.

```bash
# request
Accept-CH: RTT
# response
RTT: 125
```

# Save-Data (req)

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

# [Sec-заголовки](./sec-headers.md)

# Speculation-Rules (res) (-ff, -sf)

содержит один или несколько URL-адресов, указывающих на текстовые ресурсы, содержащие определения правил спекуляции в формате JSON. Для работы со speculations api (карта ссылок в формате json)

# Supports-Loading-Mode (res) (-ff, -sf)

Позволяет пользователю выбрать загрузку в новом, более рискованном контексте, в котором загрузка в противном случае не удалась бы.

```bash
# Ответ может быть загружен внутри изолированного фрейма. Без явного согласия на это, все навигации внутри изолированного фрейма завершатся неудачей.
```

# Supports-Loading-Mode: fenced-frame

Указывает на то, что источник данных выбирает загрузку документов посредством предварительной отрисовки на том же сайте, но с другого источника.

```bash
Supports-Loading-Mode: credentialed-prerender
```

# Use-As-Dictionary

В заголовке перечислены критерии соответствия, для которых может использоваться словарь Compression Dictionary Transport, для будущих запросов. Работает в паре с [Dictionary-ID](./deprecated-experemental.md#dictionary-id-req-res)

```bash
Use-As-Dictionary: match="<url-pattern>" # паттерн для пути который поддерживает вид словаря
Use-As-Dictionary: match-dest=("<destination1>" "<destination2>", …) # места вставки
Use-As-Dictionary: id="<string-identifier>" #  Dictionary-ID
Use-As-Dictionary: type="raw"


Content-Encoding: match="<url-pattern>", match-dest=("<destination1>")
```
