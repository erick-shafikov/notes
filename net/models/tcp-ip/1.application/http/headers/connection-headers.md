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
Access-Control-Allow-Origin: * # должен быть заголовок Vary
Access-Control-Allow-Origin: https://developer.mozilla.org

# при значении * должен быть заголовок Vary
Access-Control-Allow-Origin: https://developer.mozilla.org
Vary: Origin
```

### Access-Control-Expose-Headers

Какие заголовки ответа должны быть доступны скриптам, работающим в браузере, в ответ на междоменный запрос

### Access-Control-Max-Age

как долго живет результат предварительного запроса, значение в секундах
