sec-headers — заголовки с префиксом `Sec-`. Браузеры ограничивают возможность устанавливать их вручную (через JS или curl), что защищает сервер от подделанных запросов и даёт контекст об инициаторе.

Большинство — request-заголовки (Sec-CH-*, Sec-Fetch-*), часть — response (`Sec-WebSocket-Accept`) или req-res (`Sec-WebSocket-Extensions`, `Sec-WebSocket-Protocol`, `Sec-WebSocket-Version`).

# Sec-CH-заголовки

Client Hints — механизм, при котором клиент отправляет данные об устройстве/браузере только если сервер явно запросил их через [Accept-CH](./res-headers.md#accept-ch). Результат кешируется с учётом [Vary](./req-res-headers.md#vary).

**Low entropy** (отправляются по умолчанию без Accept-CH):
- `Sec-CH-UA`, `Sec-CH-UA-Mobile`, `Sec-CH-UA-Platform`

**High entropy** (только после Accept-CH):
- все остальные Sec-CH-* заголовки

Все заголовки относятся к экспериментальным. Большинство не работают в Firefox и Safari.

## Sec-CH-Device-Memory

Объём RAM устройства в ГБ. Значения округлены до ближайшей степени двойки для защиты приватности. Замена устаревшего [Device-Memory](./deprecated.md#device-memory-req).

```bash
# возможные значения: 0.25, 0.5, 1, 2, 4, 8
Sec-CH-Device-Memory: 4
```

```bash
# 1. сервер запрашивает данные об ОЗУ
HTTP/1.1 200 OK
Accept-CH: Sec-CH-Device-Memory
Vary: Sec-CH-Device-Memory

# 2. клиент отправляет
GET /app.js HTTP/1.1
Host: example.com
Sec-CH-Device-Memory: 0.5

# сервер отдаёт облегчённую версию для слабых устройств
HTTP/1.1 200 OK
Content-Type: text/javascript
Vary: Sec-CH-Device-Memory

(облегчённый bundle)
```

## Sec-CH-DPR

Device Pixel Ratio — отношение физических пикселей к CSS-пикселям. Используется для выбора изображения нужного разрешения. Замена устаревшего [DPR](./deprecated.md#dpr-req).

```bash
# типичные значения: 1.0 (обычный монитор), 2.0 (Retina), 3.0 (high-DPI mobile)
Sec-CH-DPR: 2.0
```

```bash
# 1. сервер запрашивает DPR
HTTP/1.1 200 OK
Accept-CH: Sec-CH-DPR
Vary: Sec-CH-DPR

# 2. клиент сообщает DPR при запросе изображения
GET /hero.jpg HTTP/1.1
Host: example.com
Sec-CH-DPR: 2.0

# сервер отдаёт изображение 2x разрешения
HTTP/1.1 200 OK
Content-Type: image/jpeg
Vary: Sec-CH-DPR
```

## Sec-CH-Prefers-Color-Scheme

Цветовая схема пользователя — light или dark. Позволяет серверу отдавать CSS/изображения, адаптированные под тему, без JavaScript.

```bash
Sec-CH-Prefers-Color-Scheme: "dark"
Sec-CH-Prefers-Color-Scheme: "light"
```

```bash
# 1. первый запрос — сервер запрашивает цветовую схему
GET / HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Content-Type: text/html
Accept-CH: Sec-CH-Prefers-Color-Scheme
Vary: Sec-CH-Prefers-Color-Scheme
Critical-CH: Sec-CH-Prefers-Color-Scheme

# 2. браузер повторяет запрос с предпочтением пользователя
GET / HTTP/1.1
Host: example.com
Sec-CH-Prefers-Color-Scheme: "dark"

HTTP/1.1 200 OK
Content-Type: text/html
Vary: Sec-CH-Prefers-Color-Scheme

(HTML с тёмной темой)
```

## Sec-CH-Prefers-Reduced-Motion

Предпочтение пользователя уменьшить анимацию (соответствует `prefers-reduced-motion` CSS). Позволяет серверу отдавать упрощённые CSS/видео без JavaScript-проверки.

```bash
Sec-CH-Prefers-Reduced-Motion: "reduce"
Sec-CH-Prefers-Reduced-Motion: "no-preference"
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-Prefers-Reduced-Motion
Vary: Sec-CH-Prefers-Reduced-Motion

GET /page HTTP/1.1
Sec-CH-Prefers-Reduced-Motion: "reduce"

HTTP/1.1 200 OK
Vary: Sec-CH-Prefers-Reduced-Motion
Content-Type: text/html

(HTML без анимации)
```

## Sec-CH-Prefers-Reduced-Transparency

Предпочтение пользователя уменьшить прозрачность (соответствует `prefers-reduced-transparency` CSS).

```bash
Sec-CH-Prefers-Reduced-Transparency: "reduce"
Sec-CH-Prefers-Reduced-Transparency: "no-preference"
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-Prefers-Reduced-Transparency
Vary: Sec-CH-Prefers-Reduced-Transparency

GET /app HTTP/1.1
Sec-CH-Prefers-Reduced-Transparency: "reduce"
```

## Sec-CH-UA

Бренд и версия браузера. **Low entropy** — отправляется автоматически без Accept-CH. Содержит список пар бренд-версия, намеренно включает ложные значения для защиты от снятия отпечатков.

```bash
# формат: "<brand>";v="<version>", ...
Sec-CH-UA: "Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"
Sec-CH-UA: "Firefox";v="121"
```

```bash
GET /page HTTP/1.1
Host: example.com
Sec-CH-UA: "Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"
Sec-CH-UA-Mobile: ?0
Sec-CH-UA-Platform: "Windows"
```

## Sec-CH-UA-Arch

Архитектура ЦПУ. High entropy — нужен Accept-CH.

```bash
# значения: "x86", "arm", ""
Sec-CH-UA-Arch: "x86"
Sec-CH-UA-Arch: "arm"
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-UA-Arch

GET /download HTTP/1.1
Sec-CH-UA-Arch: "arm"

# сервер отдаёт ARM-версию бинаря
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="app-arm64.dmg"
```

## Sec-CH-UA-Bitness

Разрядность процессора. High entropy.

```bash
# значения: "32", "64"
Sec-CH-UA-Bitness: "64"
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-UA-Bitness

GET /installer HTTP/1.1
Sec-CH-UA-Bitness: "64"
```

## Sec-CH-UA-Form-Factors

Форм-фактор устройства. High entropy.

```bash
# значения: "Desktop", "Automotive", "Mobile", "Tablet", "XR", "EInk", "Watch"
Sec-CH-UA-Form-Factors: "Desktop"
Sec-CH-UA-Form-Factors: "Mobile"
Sec-CH-UA-Form-Factors: "EInk"
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-UA-Form-Factors

GET /page HTTP/1.1
Sec-CH-UA-Mobile: ?0
Sec-CH-UA-Form-Factors: "EInk"
```

## Sec-CH-UA-Full-Version (deprecated)

Полная версия браузера. Заменён на [Sec-CH-UA-Full-Version-List](#sec-ch-ua-full-version-list).

## Sec-CH-UA-Full-Version-List

Полные версии всех брендов браузера. High entropy.

```bash
Sec-CH-UA-Full-Version-List: "Not_A Brand";v="8.0.0.0", "Chromium";v="120.0.6099.71", "Google Chrome";v="120.0.6099.71"
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-UA-Full-Version-List

GET /analytics HTTP/1.1
Sec-CH-UA-Full-Version-List: "Not_A Brand";v="8.0.0.0", "Chromium";v="120.0.6099.71"
```

## Sec-CH-UA-Mobile

Предпочитает ли пользователь мобильный UX. **Low entropy** — отправляется автоматически.

```bash
Sec-CH-UA-Mobile: ?1  # мобильное устройство или мобильный режим
Sec-CH-UA-Mobile: ?0  # не мобильное
```

```bash
GET /page HTTP/1.1
Sec-CH-UA: "Chromium";v="120"
Sec-CH-UA-Mobile: ?1
Sec-CH-UA-Platform: "Android"
```

## Sec-CH-UA-Model

Модель устройства. High entropy.

```bash
Sec-CH-UA-Model: "Pixel 7"
Sec-CH-UA-Model: "iPhone 15"
Sec-CH-UA-Model: ""  # для десктопов — пустая строка
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-UA-Model

GET /page HTTP/1.1
Sec-CH-UA-Model: "Pixel 7"
```

## Sec-CH-UA-Platform

Операционная система. **Low entropy** — отправляется автоматически.

```bash
# значения: "Android", "Chrome OS", "Chromium OS", "iOS", "Linux", "macOS", "Windows", "Unknown"
Sec-CH-UA-Platform: "Windows"
Sec-CH-UA-Platform: "macOS"
```

```bash
GET /page HTTP/1.1
Sec-CH-UA: "Chromium";v="120"
Sec-CH-UA-Mobile: ?0
Sec-CH-UA-Platform: "Windows"
```

## Sec-CH-UA-Platform-Version

Версия ОС. High entropy.

```bash
Sec-CH-UA-Platform-Version: "15.0.0"   # macOS Sequoia
Sec-CH-UA-Platform-Version: "10.0.0"   # Windows 10
Sec-CH-UA-Platform-Version: "14"       # iOS 14
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-UA-Platform-Version

GET /page HTTP/1.1
Sec-CH-UA-Platform: "Windows"
Sec-CH-UA-Platform-Version: "10.0.0"
```

## Sec-CH-UA-WoW64

Работает ли 32-битное приложение в режиме WoW64 (Windows 32-bit on 64-bit). High entropy.

```bash
Sec-CH-UA-WoW64: ?1  # да, WoW64
Sec-CH-UA-WoW64: ?0  # нет
```

## Sec-CH-Viewport-Height

Высота viewport в CSS-пикселях. High entropy.

```bash
Sec-CH-Viewport-Height: 900
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-Viewport-Height

GET /page HTTP/1.1
Sec-CH-Viewport-Height: 844   # iPhone viewport height
```

## Sec-CH-Viewport-Width

Ширина viewport в CSS-пикселях. High entropy. Замена устаревшего [Viewport-Width](./deprecated.md#viewport-width).

```bash
Sec-CH-Viewport-Width: 1280
Sec-CH-Viewport-Width: 390   # iPhone 14
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-Viewport-Width
Vary: Sec-CH-Viewport-Width

GET /hero-image.jpg HTTP/1.1
Sec-CH-Viewport-Width: 390

HTTP/1.1 200 OK
Content-Type: image/jpeg
Vary: Sec-CH-Viewport-Width
```

## Sec-CH-Width

Желаемая ширина ресурса в физических пикселях (CSS-пиксели × DPR). High entropy. Замена устаревшего [Width](./deprecated.md#width-req).

```bash
Sec-CH-Width: 780  # 390 CSS px × DPR 2.0
```

```bash
HTTP/1.1 200 OK
Accept-CH: Sec-CH-Width, Sec-CH-DPR

GET /product.jpg HTTP/1.1
Sec-CH-Width: 800
Sec-CH-DPR: 2.0
```

# Sec-заголовки

## Sec-Browsing-Topics (req)

Отправляет список интересов пользователя, вычисленных браузером локально на основе истории посещений. Используется Topics API (Privacy Sandbox) — замена сторонних куки для таргетированной рекламы. Требует включения Topics API и явного разрешения через Permissions-Policy.

```bash
Sec-Browsing-Topics: ();p=P000000000
Sec-Browsing-Topics: (1 2 3);v=chrome.1:1:2, ();p=P000000000
```

```bash
# запрос к рекламному серверу с темами пользователя
GET /ad HTTP/1.1
Host: adtech.example
Sec-Browsing-Topics: (6 8 23);v=chrome.1:1:2, ();p=P000000000

HTTP/1.1 200 OK
Observe-Browsing-Topics: ?1   # сервер регистрирует посещение
Content-Type: text/html
```

## Fetch metadata

Группа заголовков запроса, которые браузер добавляет автоматически. Дают серверу контекст о том, откуда и зачем инициирован запрос, без возможности подделки со стороны JS.

### Sec-Fetch-Dest

Назначение запроса — куда будут использованы полученные данные.

```bash
# значения:
Sec-Fetch-Dest: document     # основной HTML-документ
Sec-Fetch-Dest: script       # <script src="...">
Sec-Fetch-Dest: style        # <link rel="stylesheet">
Sec-Fetch-Dest: image        # <img src="...">
Sec-Fetch-Dest: font         # @font-face
Sec-Fetch-Dest: fetch        # fetch() / XMLHttpRequest
Sec-Fetch-Dest: iframe       # <iframe>
Sec-Fetch-Dest: audio        # <audio>
Sec-Fetch-Dest: video        # <video>
Sec-Fetch-Dest: worker       # new Worker()
Sec-Fetch-Dest: serviceworker  # регистрация Service Worker
Sec-Fetch-Dest: manifest     # Web App Manifest
Sec-Fetch-Dest: report       # отчёт о нарушениях (CSP, NEL)
Sec-Fetch-Dest: empty        # fetch() без явного назначения
```

### Sec-Fetch-Mode

Режим запроса — как браузер обрабатывает CORS.

```bash
Sec-Fetch-Mode: navigate      # навигация верхнего уровня (переход по ссылке)
Sec-Fetch-Mode: cors          # CORS-запрос (fetch с cross-origin)
Sec-Fetch-Mode: no-cors       # no-cors запрос (image, script без CORS)
Sec-Fetch-Mode: same-origin   # ресурс с того же origin
Sec-Fetch-Mode: websocket     # WebSocket handshake
```

### Sec-Fetch-Site

Отношение между origin инициатора запроса и origin запрашиваемого ресурса.

```bash
Sec-Fetch-Site: same-origin   # тот же origin (scheme + host + port)
Sec-Fetch-Site: same-site     # тот же сайт (eTLD+1), но возможно разный origin
Sec-Fetch-Site: cross-site    # другой сайт
Sec-Fetch-Site: none          # инициатор отсутствует (прямой переход, закладки)
```

### Sec-Fetch-Storage-Access

Статус доступа к хранилищу в cross-site контексте. Используется в связке с [Activate-Storage-Access](./res-headers.md#activate-storage-access).

```bash
Sec-Fetch-Storage-Access: none      # разрешения нет
Sec-Fetch-Storage-Access: inactive  # разрешение есть, но не активировано
Sec-Fetch-Storage-Access: active    # разрешение активировано, куки включены
```

### Sec-Fetch-User (-sf)

Отправляется только при запросах, инициированных действием пользователя (клик по ссылке, submit формы). Всегда равен `?1`.

```bash
Sec-Fetch-User: ?1
```

**Диалог Fetch metadata — все четыре заголовка вместе:**

```bash
# навигация по ссылке (пользователь кликает)
GET /page HTTP/1.1
Host: example.com
Sec-Fetch-Site: none
Sec-Fetch-Mode: navigate
Sec-Fetch-Dest: document
Sec-Fetch-User: ?1

# JS-запрос с той же страницы к стороннему API
GET /api/data HTTP/1.1
Host: api.other.com
Sec-Fetch-Site: cross-site
Sec-Fetch-Mode: cors
Sec-Fetch-Dest: empty

# загрузка картинки со стороннего CDN
GET /logo.png HTTP/1.1
Host: cdn.example.com
Sec-Fetch-Site: cross-site
Sec-Fetch-Mode: no-cors
Sec-Fetch-Dest: image

# iframe загружает свой документ
GET /widget HTTP/1.1
Host: widget.example.com
Sec-Fetch-Site: cross-site
Sec-Fetch-Mode: navigate
Sec-Fetch-Dest: iframe
```

**Пример защиты на сервере с помощью Fetch metadata:**

```python
def protect(request):
    # разрешить навигацию и same-origin
    if request.headers.get("Sec-Fetch-Site") in ("same-origin", "none"):
        return handle(request)
    # заблокировать cross-site запросы к API
    if request.headers.get("Sec-Fetch-Mode") == "navigate":
        return handle(request)
    return Response(403)
```

## Sec-GPC (+ff)

Global Privacy Control — пользователь запрещает продажу и передачу своих данных третьим лицам. Браузерная замена устаревшего [DNT](./deprecated.md#dnt-req). Поддерживается Firefox и некоторыми браузерами с расширениями.

```bash
Sec-GPC: 1  # пользователь запрещает передачу данных
```

```bash
GET /page HTTP/1.1
Host: example.com
Sec-GPC: 1

HTTP/1.1 200 OK
Content-Type: text/html
# сервер обязан учесть предпочтение (в юрисдикциях, где GPC имеет юридическую силу)
```

## Sec-Private-State заголовки

Заголовки Private State Token API — механизм антифрод-проверки без отслеживания пользователя. Эмитент выдаёт токены (криптографические), редемпция подтверждает что запрос сделан реальным пользователем.

### Sec-Private-State-Token (req-res)

Содержит токен для операций выпуска (issuance) и погашения (redemption) в Private State Token API.

```bash
# запрос на выпуск токена (issuance)
POST /issue HTTP/1.1
Host: issuer.example
Sec-Private-State-Token: <blinded-token-data>

# ответ эмитента
HTTP/1.1 200 OK
Sec-Private-State-Token: <signed-token-data>

# запрос на погашение (redemption)
POST /redeem HTTP/1.1
Host: issuer.example
Sec-Private-State-Token: <token>

HTTP/1.1 200 OK
Sec-Private-State-Token: <redemption-record>
```

### Sec-Private-State-Token-Crypto-Version

Криптографический протокол для Private State Token операций.

```bash
Sec-Private-State-Token-Crypto-Version: PrivacyPass-v3
```

### Sec-Private-State-Token-Lifetime (res)

Срок жизни записи о погашении токена в секундах.

```bash
Sec-Private-State-Token-Lifetime: 86400  # 24 часа
```

## Sec-Purpose

Указывает цель запроса, если она отличается от прямого использования браузером (например, prefetch для будущей навигации).

```bash
Sec-Purpose: prefetch           # ресурс prefetch-ируется для будущего использования
Sec-Purpose: prefetch;prerender # ресурс для prerender
```

```bash
# браузер prefetch-ирует страницу по speculation rules
GET /next-page HTTP/1.1
Host: example.com
Sec-Purpose: prefetch
Sec-Fetch-Dest: document
Sec-Fetch-Mode: navigate

HTTP/1.1 200 OK
Content-Type: text/html
# сервер может вернуть облегчённую версию или отказать в prefetch
```

## Sec-Redemption-Record

Содержит записи о погашении Private State Token — список пар «эмитент + запись о погашении». Прикрепляется к запросу для подтверждения что пользователь ранее прошёл проверку у эмитента.

```bash
Sec-Redemption-Record: issuer.example:AQIDBAUGBwg..., other-issuer.example:BQIDBAUG...
```

## Sec-Speculation-Tags (-ff, -sf)

Тег правила из `<script type="speculationrules">`, которое инициировало prefetch или prerender. Позволяет серверу определить какое именно правило вызвало запрос и при необходимости заблокировать его.

```bash
Sec-Speculation-Tags: null       # тег не задан (правило без тега)
Sec-Speculation-Tags: "my-rule"  # тег задан в правиле
```

```html
<script type="speculationrules">
{
  "prefetch": [{
    "urls": ["next.html"],
    "tag": "my-rule"
  }]
}
</script>
```

```bash
# браузер prefetch-ирует next.html по правилу "my-rule"
GET /next.html HTTP/1.1
Sec-Speculation-Tags: "my-rule"
Sec-Purpose: prefetch

# сервер может заблокировать конкретное правило
HTTP/1.1 200 OK
```

## Sec-WebSocket

Заголовки WebSocket handshake — HTTP/1.1 используется только для установки соединения (`101 Switching Protocols`), после чего происходит переключение на WebSocket-протокол.

### Sec-WebSocket-Key (req)

Случайный 16-байтный base64-ключ, генерируемый клиентом. Сервер использует его для формирования [Sec-WebSocket-Accept](#sec-websocket-accept-res).

```bash
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
```

### Sec-WebSocket-Accept (res)

Подтверждение handshake со стороны сервера. Вычисляется как `base64(SHA-1(Sec-WebSocket-Key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))`. Клиент проверяет значение перед переключением на WebSocket.

```bash
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

### Sec-WebSocket-Version (req-res)

Версия протокола WebSocket. Клиент указывает желаемую версию (всегда `13` по RFC 6455). Если сервер не поддерживает — возвращает список поддерживаемых версий.

```bash
# клиент
Sec-WebSocket-Version: 13

# сервер не поддерживает — 426 Upgrade Required
HTTP/1.1 426 Upgrade Required
Sec-WebSocket-Version: 13
```

### Sec-WebSocket-Protocol (req-res)

Подпротокол WebSocket — протокол прикладного уровня поверх WebSocket. Клиент предлагает список, сервер выбирает один.

```bash
# клиент предлагает несколько
Sec-WebSocket-Protocol: soap, wamp, chat

# сервер выбирает один
Sec-WebSocket-Protocol: chat
```

### Sec-WebSocket-Extensions (req-res)

Расширения WebSocket — переговоры о сжатии и других опциях. Клиент предлагает расширения, сервер подтверждает поддерживаемые.

```bash
# клиент предлагает permessage-deflate (сжатие)
Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits

# сервер подтверждает с параметрами
Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits=15
```

**Полный WebSocket handshake:**

```bash
# клиент инициирует upgrade
GET /chat HTTP/1.1
Host: example.com
Connection: Upgrade
Upgrade: websocket
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
Sec-WebSocket-Protocol: chat, superchat
Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits

# сервер подтверждает — соединение установлено
HTTP/1.1 101 Switching Protocols
Connection: Upgrade
Upgrade: websocket
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
Sec-WebSocket-Protocol: chat
Sec-WebSocket-Extensions: permessage-deflate; client_max_window_bits=15

# далее идёт WebSocket-фрейминг
```
