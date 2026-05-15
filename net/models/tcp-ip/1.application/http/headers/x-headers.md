Заголовки с префиксом `X-` — исторически неофициальные/кастомные HTTP-заголовки. Если заголовок становился популярным — префикс убирали и стандартизировали (например, `X-Forwarded-For` → `Forwarded`). Сегодня префикс `X-` считается legacy-подходом (RFC 6648). При использовании в `fetch()` браузер отправляет preflight, так как это non-simple заголовок.

# X-Content-Type-Options (res)

Запрещает браузеру выполнять MIME-сниффинг — угадывание типа содержимого по его содержимому вместо объявленного [Content-Type](./representation-headers.md#content-type). Без этого заголовка браузер может интерпретировать `text/plain` как JavaScript и выполнить его — это открывает вектор XSS.

Единственное допустимое значение — `nosniff`.

```bash
X-Content-Type-Options: nosniff
```

```bash
# без заголовка — браузер может "угадать" тип
GET /upload/file.txt HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Content-Type: text/plain

<script>alert(1)</script>
# ⚠ браузер может выполнить как HTML/JS

# с заголовком — браузер строго соблюдает Content-Type
HTTP/1.1 200 OK
Content-Type: text/plain
X-Content-Type-Options: nosniff

<script>alert(1)</script>
# ✓ браузер отображает как текст, не выполняет
```

```bash
# типичное использование — на всех страницах/ресурсах
HTTP/1.1 200 OK
Content-Type: application/json
X-Content-Type-Options: nosniff
Content-Length: 42

{"user": "alice"}
```

# X-DNS-Prefetch-Control (res)

Управляет DNS-предзагрузкой для ссылок на странице. Браузеры по умолчанию заранее разрешают DNS для ссылок, чтобы ускорить навигацию. Этот заголовок позволяет явно включить или отключить поведение — например, для сайтов с повышенными требованиями к приватности.

```bash
X-DNS-Prefetch-Control: on   # разрешить DNS prefetch (по умолчанию для HTTP)
X-DNS-Prefetch-Control: off  # запретить (по умолчанию для HTTPS)
```

```bash
# сервер запрещает DNS prefetch (банк, мед. сайт — утечка ссылок нежелательна)
GET /account HTTP/1.1
Host: bank.example.com

HTTP/1.1 200 OK
Content-Type: text/html
X-DNS-Prefetch-Control: off

(html с ссылками на партнёрские сайты — браузер не будет резолвить их DNS заранее)
```

Для точечного управления на уровне страницы — HTML:

```html
<meta http-equiv="x-dns-prefetch-control" content="off" />
<link rel="dns-prefetch" href="https://cdn.example.com" />
```

# X-Forwarded-For (req)

Передаёт цепочку IP-адресов: исходный IP клиента и все промежуточные прокси. Добавляется каждым прокси при пересылке запроса. **Может быть подделан клиентом** — не использовать для аутентификации без дополнительной проверки. Стандартизированная замена — [Forwarded](./req-headers.md#forwarded).

```bash
# X-Forwarded-For: <client-ip>, <proxy1-ip>, <proxy2-ip>
X-Forwarded-For: 203.0.113.195
X-Forwarded-For: 203.0.113.195, 70.41.3.18
X-Forwarded-For: 203.0.113.195, 70.41.3.18, 150.172.238.178
```

```bash
# клиент (203.0.113.195) → proxy-a (70.41.3.18) → proxy-b → origin

# proxy-a получает запрос от клиента и добавляет его IP
GET /page HTTP/1.1
Host: origin.example.com
X-Forwarded-For: 203.0.113.195

# proxy-b добавляет IP proxy-a в конец
GET /page HTTP/1.1
Host: origin.example.com
X-Forwarded-For: 203.0.113.195, 70.41.3.18

# origin видит реальный IP клиента в первом значении
HTTP/1.1 200 OK
Content-Type: text/html
```

# X-Forwarded-Host (req)

Передаёт оригинальный заголовок `Host`, который клиент отправил прокси или балансировщику нагрузки. Полезен когда прокси перезаписывает `Host` для роутинга на бэкенд. Стандартизированная замена — [Forwarded](./req-headers.md#forwarded).

```bash
X-Forwarded-Host: example.com
X-Forwarded-Host: example.com:443
```

```bash
# клиент запрашивает example.com, балансировщик проксирует на internal-backend
GET /page HTTP/1.1
Host: internal-backend.local        # перезаписан балансировщиком
X-Forwarded-Host: example.com       # оригинальный Host от клиента
X-Forwarded-For: 203.0.113.195
X-Forwarded-Proto: https

HTTP/1.1 200 OK
Content-Type: text/html
```

# X-Forwarded-Proto (req)

Передаёт протокол (HTTP или HTTPS), который клиент использовал при подключении к прокси или балансировщику нагрузки. Бэкенд обычно работает по HTTP, но должен знать с какого протокола пришёл клиент — для генерации корректных redirect-URL. Стандартизированная замена — [Forwarded](./req-headers.md#forwarded).

```bash
X-Forwarded-Proto: https
X-Forwarded-Proto: http
```

```bash
# клиент подключился по HTTPS, балансировщик пересылает по HTTP на бэкенд
GET /account HTTP/1.1
Host: backend.internal
X-Forwarded-Proto: https
X-Forwarded-For: 203.0.113.195
X-Forwarded-Host: example.com

# бэкенд генерирует redirect с учётом оригинального протокола
HTTP/1.1 302 Found
Location: https://example.com/account/dashboard
```

# X-Frame-Options (res)

Управляет возможностью встраивать страницу через `<frame>`, `<iframe>`, `<embed>`, `<object>`. Без этого заголовка любой сайт может встроить страницу и использовать её для clickjacking-атак. Предпочтительная замена — директива `frame-ancestors` в [Content-Security-Policy](./res-headers.md#content-security-policy).

```bash
X-Frame-Options: DENY        # запретить встраивание отовсюду
X-Frame-Options: SAMEORIGIN  # разрешить встраивание только с того же origin
```

```bash
# защита страницы входа от clickjacking
GET /login HTTP/1.1
Host: bank.example.com

HTTP/1.1 200 OK
Content-Type: text/html
X-Frame-Options: DENY

(страница логина)
# любая попытка встроить в <iframe> на другом сайте будет заблокирована браузером
```

```bash
# разрешить встраивание только внутри своего сайта
HTTP/1.1 200 OK
X-Frame-Options: SAMEORIGIN
Content-Type: text/html
```

```bash
# современный эквивалент через CSP (предпочтительнее)
HTTP/1.1 200 OK
Content-Security-Policy: frame-ancestors 'none'          # = DENY
Content-Security-Policy: frame-ancestors 'self'          # = SAMEORIGIN
Content-Security-Policy: frame-ancestors https://partner.example.com  # точный origin
```

# X-Permitted-Cross-Domain-Policies (res)

Определяет метаполитику доступа к ресурсам сайта для клиентов Adobe Flash и Adobe Acrobat. Используется когда `crossdomain.xml` не размещён в корне или его поведение нужно ограничить. Актуален только для окружений с legacy Flash/PDF-плагинами.

```bash
X-Permitted-Cross-Domain-Policies: none           # запретить все cross-domain запросы Flash/PDF
X-Permitted-Cross-Domain-Policies: master-only    # только из root crossdomain.xml
X-Permitted-Cross-Domain-Policies: by-content-type  # только XML с Content-Type: text/x-cross-domain-policy
X-Permitted-Cross-Domain-Policies: all            # разрешить все policy-файлы
```

```bash
# рекомендуемая безопасная настройка
HTTP/1.1 200 OK
X-Permitted-Cross-Domain-Policies: none
Content-Type: text/html
```

# X-Powered-By (res)

Идентифицирует серверное ПО/фреймворк, сгенерировавший ответ. Добавляется автоматически многими фреймворками (Express, PHP, ASP.NET). **Рекомендуется отключить** — раскрытие версий облегчает поиск уязвимостей.

```bash
X-Powered-By: Express
X-Powered-By: PHP/8.2.0
X-Powered-By: ASP.NET
```

```bash
# типичный ответ с включённым заголовком
HTTP/1.1 200 OK
X-Powered-By: Express
Content-Type: application/json

{"status": "ok"}

# в Express отключается так:
# app.disable('x-powered-by')
```

# X-Robots-Tag (res)

Управляет индексацией страницы или ресурса поисковыми краулерами — HTTP-аналог мета-тега `<meta name="robots">`. Применим к любому типу контента (PDF, изображения), не только к HTML.

```bash
# X-Robots-Tag: <indexing-rule>
# X-Robots-Tag: <bot-name>: <indexing-rule>
# X-Robots-Tag: <indexing-rule>, <bot-name>: <indexing-rule>
```

Директивы:

```bash
all              # без ограничений (по умолчанию)
noindex          # не показывать в результатах поиска
nofollow         # не переходить по ссылкам на странице
none             # noindex + nofollow
nosnippet        # без текстового сниппета и превью видео
indexifembedded  # индексировать если встроена через iframe (работает только с noindex)
notranslate      # не предлагать перевод
noimageindex     # не индексировать изображения на странице
max-snippet: N   # сниппет не длиннее N символов
max-image-preview: none | standard | large
max-video-preview: 0 | -1
unavailable_after: <date>  # не показывать после указанной даты
```

```bash
# запрет индексации страницы с личными данными
GET /user/123/profile HTTP/1.1
Host: example.com

HTTP/1.1 200 OK
Content-Type: text/html
X-Robots-Tag: noindex, nofollow
```

```bash
# запрет только для конкретного бота, остальные индексируют
HTTP/1.1 200 OK
Content-Type: application/pdf
X-Robots-Tag: googlebot: noindex
X-Robots-Tag: all
```

```bash
# ограничить сниппет до 160 символов и превью изображения
HTTP/1.1 200 OK
Content-Type: text/html
X-Robots-Tag: max-snippet: 160, max-image-preview: standard
```

```bash
# страница доступна для индексации только до определённой даты
HTTP/1.1 200 OK
Content-Type: text/html
X-Robots-Tag: unavailable_after: 2025-12-31T00:00:00Z
```

# X-XSS-Protection (res) (deprecated)

Включал встроенный XSS-фильтр в Internet Explorer, старых Chrome и Safari. Заголовок устарел и удалён из современных браузеров. Рекомендуется использовать [Content-Security-Policy](./res-headers.md#content-security-policy) с директивой `script-src`.

```bash
X-XSS-Protection: 0              # фильтр выключен
X-XSS-Protection: 1              # фильтр включён — браузер удаляет небезопасные части
X-XSS-Protection: 1; mode=block  # заблокировать страницу полностью при обнаружении XSS
X-XSS-Protection: 1; report=<reporting-uri>  # включить и отправить отчёт
```

```bash
# современная замена
HTTP/1.1 200 OK
Content-Security-Policy: default-src 'self'; script-src 'self'
X-XSS-Protection: 0  # явно отключить legacy-фильтр чтобы избежать конфликтов
Content-Type: text/html
```
