заголовки OPTIONS (предварительного preflight-запроса)

Preflight — автоматический запрос браузера методом OPTIONS перед «несимпличным» cross-origin запросом. Браузер проверяет, разрешит ли сервер основной запрос, и только при положительном ответе его отправляет.

**Простые запросы** (preflight не нужен):

- методы: GET, HEAD, POST
- заголовки: только Accept, Accept-Language, Content-Language, Content-Type (только `text/plain`, `multipart/form-data`, `application/x-www-form-urlencoded`)

**Несимпличные запросы** (вызывают preflight):

- методы: PUT, DELETE, PATCH и другие
- нестандартные заголовки: `Authorization`, `X-Custom-Header`, `Content-Type: application/json`

Полный диалог preflight → основной запрос:

```bash
# 1. браузер автоматически отправляет OPTIONS
OPTIONS /api/resource HTTP/1.1
Host: api.example.com
Origin: https://app.example.com
Access-Control-Request-Method: DELETE
Access-Control-Request-Headers: Content-Type, Authorization

# 2. сервер разрешает запрос
HTTP/1.1 204 No Content
Access-Control-Allow-Origin: https://app.example.com
Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Max-Age: 86400
Vary: Origin

# 3. браузер отправляет основной запрос
DELETE /api/resource HTTP/1.1
Host: api.example.com
Origin: https://app.example.com
Content-Type: application/json
Authorization: Bearer eyJhbGci...

# 4. сервер отвечает на основной запрос
HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://app.example.com
```

# заголовки предварительного запроса

## Access-Control-Request-Headers

Браузер перечисляет заголовки, которые будут отправлены в основном запросе. Сервер должен разрешить их в [Access-Control-Allow-Headers](#access-control-allow-headers).

```bash
# браузер сообщает что основной запрос будет содержать эти заголовки
Access-Control-Request-Headers: Content-Type, Authorization, X-Custom-Header
```

## Access-Control-Request-Method

Браузер сообщает метод основного запроса. Сервер должен разрешить его в [Access-Control-Allow-Methods](#access-control-allow-methods).

```bash
# браузер сообщает что основной запрос будет DELETE
Access-Control-Request-Method: DELETE
```

# заголовки предварительного ответа

## Access-Control-Allow-Credentials

Разрешает браузеру передавать и читать credentials (cookies, HTTP-аутентификация, TLS-сертификаты) в cross-origin запросах.

**Ключевое ограничение**: при `credentials: "include"` в `Access-Control-Allow-Origin` нельзя использовать `*` — только конкретный origin.

```bash
# правильно: конкретный origin + credentials: true
Access-Control-Allow-Origin: https://app.example.com
Access-Control-Allow-Credentials: true

# неправильно — браузер заблокирует ответ
Access-Control-Allow-Origin: *
Access-Control-Allow-Credentials: true
```

```bash
# полный диалог с credentials
OPTIONS /api/user HTTP/1.1
Origin: https://app.example.com
Access-Control-Request-Method: GET

HTTP/1.1 204 No Content
Access-Control-Allow-Origin: https://app.example.com
Access-Control-Allow-Credentials: true

# основной запрос с cookies
GET /api/user HTTP/1.1
Origin: https://app.example.com
Cookie: session=abc123

HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://app.example.com
Access-Control-Allow-Credentials: true
```

```js
fetch(url, { credentials: "include" });
```

## Access-Control-Allow-Headers

Перечисляет заголовки, которые разрешено использовать в основном запросе. Должен содержать все заголовки из `Access-Control-Request-Headers`.

```bash
Access-Control-Allow-Headers: Content-Type, Authorization, X-Custom-Header

# разрешить любые заголовки
Access-Control-Allow-Headers: *
```

```bash
# OPTIONS запрос
OPTIONS /resource/foo HTTP/1.1
Origin: https://www.example.com
Access-Control-Request-Method: GET
Access-Control-Request-Headers: Content-Type, X-Requested-With

# ответ сервера
HTTP/1.1 204 No Content
Access-Control-Allow-Origin: https://www.example.com
Access-Control-Allow-Methods: POST, GET, OPTIONS, DELETE
Access-Control-Allow-Headers: Content-Type, X-Requested-With
Access-Control-Max-Age: 86400
```

## Access-Control-Allow-Methods

Перечисляет HTTP-методы, разрешённые для cross-origin запросов к ресурсу.

```bash
Access-Control-Allow-Methods: GET, POST, OPTIONS

# разрешить все методы
Access-Control-Allow-Methods: *
```

## Access-Control-Allow-Origin

Указывает какой origin (или все) может получить доступ к ресурсу. При динамическом значении нужен `Vary: Origin`, чтобы кеш не отдал ответ с `Origin: A` другому запросу с `Origin: B`.

```bash
# разрешить всем (нельзя комбинировать с Allow-Credentials: true)
Access-Control-Allow-Origin: *

# разрешить конкретному origin
Access-Control-Allow-Origin: https://developer.mozilla.org
Vary: Origin
```

```bash
# полный cross-origin диалог без credentials
GET /data.json HTTP/1.1
Origin: https://app.example.com

HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://app.example.com
Vary: Origin
Content-Type: application/json
```

## Access-Control-Expose-Headers

По умолчанию браузер открывает JS только базовые заголовки ответа: `Cache-Control`, `Content-Language`, `Content-Length`, `Content-Type`, `Expires`, `Last-Modified`, `Pragma`. Этот заголовок расширяет список доступных заголовков.

```bash
# открыть X-Request-Id и X-Rate-Limit для JS
Access-Control-Expose-Headers: X-Request-Id, X-Rate-Limit

# открыть все
Access-Control-Expose-Headers: *
```

```js
fetch("https://api.example.com/data").then((res) => {
  console.log(res.headers.get("X-Request-Id")); // доступен
  console.log(res.headers.get("X-Rate-Limit")); // доступен
  console.log(res.headers.get("X-Hidden-Header")); // null — не в списке
});
```

## Access-Control-Max-Age

Сколько секунд браузер кеширует результат preflight-запроса. В течение этого времени браузер не будет отправлять повторный OPTIONS перед теми же запросами.

```bash
# кешировать preflight на 24 часа
Access-Control-Max-Age: 86400

# отключить кеширование preflight
Access-Control-Max-Age: -1
```

Максимальные значения: Firefox — 86400 с (24 ч), Chromium — 7200 с (2 ч).

# Max-Forwards (req)

Используется с TRACE и OPTIONS. Ограничивает количество узлов (прокси, шлюзов), через которые может пройти запрос. Каждый промежуточный узел уменьшает значение на 1. При достижении 0 узел отвечает сам, не пересылая запрос дальше.

```bash
# TRACE с ограничением в 3 узла
TRACE /index.html HTTP/1.1
Host: example.com
Max-Forwards: 3

# узел 1 (proxy-a) — уменьшает и пересылает
TRACE /index.html HTTP/1.1
Via: 1.1 proxy-a
Max-Forwards: 2

# узел 2 (proxy-b)
TRACE /index.html HTTP/1.1
Via: 1.1 proxy-a, 1.1 proxy-b
Max-Forwards: 1

# узел 3 достигает 0 — отвечает сам
HTTP/1.1 200 OK
Content-Type: message/http

TRACE /index.html HTTP/1.1
Via: 1.1 proxy-a, 1.1 proxy-b
Max-Forwards: 0
```
