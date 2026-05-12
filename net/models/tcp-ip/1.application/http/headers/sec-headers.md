sec-headers - secure заголовки, которые браузер подставляет сам. Цель - защитить сервер от подделанных запросов (например, через curl, Postman или вредоносные скрипты) и дать серверу больше контекста о том, как именно был инициирован запрос. Браузеры ограничивают возможность вручную устанавливать эти заголовки. Все относятся к request-заголовкам. Браузер начнет отправлять ch-заголовки, если сервер ответил [Accept-CH](./res-headers.md#accept-ch)

# Sec-CH-заголовки

CH - client hints заголовки которые отравляет клиента если от сервера пришел Accept-CH заголовок в ответе, может комбинироваться с Vary
Все заголовки относятся к экспериментальным. не работают в ff и sf

## Sec-CH-Device-Memory

device client hints определяет доступное количество RAM, часть Device Memory API. Сервер должен реагировать на этот заголовок с помощью [Accept-CH](./res-headers.md#accept-ch) Значения дублируются в [Vary](!!!TODO) что бы указать значение при кеширование

```bash
# сервер отправляет
Accept-CH: Sec-CH-Device-Memory
# клиент отправляет
Sec-CH-Device-Memory: 1
```

## Sec-CH-DPR

device client hints определяющая DPR клиента, используется при отправки изображения

```bash
# сервер отправляет
Accept-CH: Sec-CH-DPR
# клиент отправляет
Sec-CH-Device-Memory: 2.0
```

## Sec-CH-Prefers-Color-Scheme

device client hints для цветовой гаммы

```bash
# запрос клиента
GET / HTTP/1.1
Host: example.com

# HTTP/1.1 200 OK
Content-Type: text/html
Accept-CH: Sec-CH-Prefers-Color-Scheme
Vary: Sec-CH-Prefers-Color-Scheme
Critical-CH: Sec-CH-Prefers-Color-Scheme

# запрос клиента
GET / HTTP/1.1
Host: example.com
Sec-CH-Prefers-Color-Scheme: "dark"
```

## Sec-CH-Prefers-Reduced-Motion

agent client hint для Reduced-Motion, механизм такой же

## Sec-CH-Prefers-Reduced-Transparency

user agent client hint для prefers-reduced-transparency, механизм такой же

## Sec-CH-UA

это подсказка для клиента пользовательского агента, содержащая информацию о его бренде и версиях.

```bash
Sec-CH-UA: " Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"
```

## Sec-CH-UA-Arch

для подсказки архитектуры цпу

```bash
GET /my/page HTTP/1.1
Host: example.site

Sec-CH-UA: " Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"
Sec-CH-UA-Mobile: ?0
Sec-CH-UA-Platform: "Windows"
Sec-CH-UA-Arch: "x86"
```

## Sec-CH-UA-Bitness

подсказка для разрядности процессора (32, 64)

## Sec-CH-UA-Form-Factors

подсказка о фром-факторе

```bash
Sec-CH-UA-Mobile: ?0
Sec-CH-UA-Form-Factors: "EInk" # "Desktop", "Automotive", "Mobile", "Tablet", "XR", "EInk", "Watch"
```

## Sec-CH-UA-Full-Version (deprecated)

подсказка полной версии

## Sec-CH-UA-Full-Version-List

подсказка полной версии

```bash
Sec-CH-UA-Full-Version-List: " Not A;Brand";v="99.0.0.0", "Chromium";v="98.0.4750.0", "Google Chrome";v="98.0.4750.0"
```

## Sec-CH-UA-Mobile

подсказка что сайт будет открываться на мобильном устройстве

```bash
Sec-CH-UA-Mobile: ?1 #?0
```

## Sec-CH-UA-Model

для определения модели

```bash
Sec-CH-UA-Model: "Pixel 3 XL"
```

## Sec-CH-UA-Platform

для определения платформы

```bash
Sec-CH-UA-Platform: "macOS" # "Android", "Chrome OS", "Chromium OS", "iOS", "Linux", "macOS", "Windows", or "Unknown"
```

## Sec-CH-UA-Platform-Version

версия платформы

```bash
Sec-CH-UA-Platform: "Windows"
Sec-CH-UA-Platform-Version: "10.0.0"
```

## Sec-CH-UA-WoW64

подсказка если 32-битное приложение запущено на 64-битной windows машине

## Sec-CH-Viewport-Height

высота экрана в css-пикселях

## Sec-CH-Viewport-Width

ширина экрана в css-пикселях

## Sec-CH-Width

ширина экрана в физических единицах

# Sec-заголовки

## Sec-Browsing-Topics (req)

Отправляет технически параметры браузера, на основе которых будет выбрана персонализированная реклама. Должен быть включен Topics API

## Fetch metadata

группа заголовков HTTP-запроса, которые предоставляют серверу информацию о контексте, в котором выполняется запрос

### Sec-Fetch-Dest

указывает на место назначения запроса. Это инициатор исходного запроса на получение данных, то есть куда (и как) будут использоваться полученные данные.

```bash
Sec-Fetch-Dest: audio # audioworklet, document, embed, empty, fencedframe, font, frame, iframe, image, json, manifest, object, paintworklet, report, script, serviceworker, sharedworker, style, track, video, webidentity, worker, xslt
```

### Sec-Fetch-Mode

Заголовок запроса HTTP Sec-Fetch-Mode fetch metadata указывает режим запроса.

```bash
Sec-Fetch-Mode: cors # navigate, no-cors, same-origin, websocket
# при навигации браузер отдаст
Sec-Fetch-Dest: document
Sec-Fetch-Mode: navigate
Sec-Fetch-Site: same-origin
Sec-Fetch-User: ?1
# при загрузке картинки
Sec-Fetch-Dest: image
Sec-Fetch-Mode: no-cors
Sec-Fetch-Site: cross-site
```

### Sec-Fetch-Site

указывает на связь между источником инициатора запроса и источником запрашиваемого ресурса.

### Sec-Fetch-Storage-Access

предоставляет "статус доступа к хранилищу" для текущего контекста выборки.

### Sec-Fetch-User (-sf)

отправляется для запросов, инициированных активацией пользователя, и его значение всегда равно ?1.

## Sec-GPC (+ff)

часть Global Privacy Control апи управляет тем, дает ли пользователь согласие на продажу или передачу его персональных данных третьим лицам веб-сайтом или сервисом.

## Sec-Private-State заголовки

заголовки Private State Token API

### Sec-Private-State-Token (req-res)

Значение токена управляет генерацией необходимых данных для запросов и ответов на операции выпуска и погашения токенов в закрытом состояния

### Sec-Private-State-Token-Crypto-Version

управление криптографическим протоколом

### Sec-Private-State-Token-Lifetime (res)

как долго запись о погашении будет существовать

## Sec-Purpose

указывает на цель использования запрашиваемого ресурса, если эта цель не связана с непосредственным использованием пользовательским агентом.

```bash
Sec-Purpose: prefetch
```

## Sec-Redemption-Record

при пересылке записей о погашении. Заголовок содержит список пар «эмитент и запись о погашении», соответствующих каждой записи о погашении.

## Sec-Speculation-Tags (-ff, -sf)

содержит одно или несколько значений тегов [из правил спекуляции script type="speculationrules"](../../../../../../fe/html/tags/meta-tags.md#script), которые привели к возникновению спекуляции. Это позволяет серверу определить, какое(ие) правило(а) вызвало(ли) спекуляцию, и потенциально заблокировать их.

```html
<script type="speculationrules">
  {
    "prefetch": [
      {
        "urls": ["next.html", "next2.html"]
      }
    ]
  }
</script>
```

```bash
Sec-Speculation-Tags: null # отключение
Sec-Speculation-Tags: "my-rule"
```

## Sec-WebSocket

### Sec-WebSocket-Accept (res)

используется в процессе установления соединения WebSocket для указания того, что сервер готов перейти к соединению WebSocket. В ответ приходит один раз и работает в паре с Sec-WebSocket-Key

```bash
Sec-WebSocket-Accept: <hashed key>
```

### Sec-WebSocket-Extensions (res+req)

используются в процессе установления соединения WebSocket для согласования расширения протокола, используемого клиентом и сервером.

### Sec-WebSocket-Key

используется в процессе установления соединения WebSocket, чтобы позволить клиенту (агенту пользователя) подтвердить, что он «действительно хочет» запросить преобразование HTTP-клиента в WebSocket.

```bash
# запрос на установку соединения
GET /chat HTTP/1.1
Host: example.com:8000
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

# ответ на базе кодирования sha-1
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

### Sec-WebSocket-Protocol (res+req)

используются в процессе установления соединения WebSocket для согласования подпротокола (soap, wamp), который будет использоваться в процессе связи.

### Sec-WebSocket-Version (res+req)

используется в процессе установления соединения WebSocket для указания поддерживаемого клиентом протокола WebSocket, а также версий протокола, поддерживаемых сервером, если он не поддерживает версию, указанную в запросе.
