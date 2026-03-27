# заголовки запроса

Заголовки по типу запроса клиента:

## Accept

клиент сообщает какие типы понимает. какие типы контента MIME может понять, сервер возвращая Content-Type сервер сообщает какой тип отправил:

```bash
Accept: text/html, application/xhtml+xml, application/xml;q=0.9, _/_;q=0.8
```

## Accept-Language

какой язык предпочитает клиент, это те же значения что и генерирует navigator.languages. Сервер может вернуть 406. Пример:

```bash
Accept-Language: fr-CH, fr;q=0.9, en;q=0.8, de;q=0.7, *;q=0.5
# или
Accept-Language: de
```

## Alt-Used

сообщает какой альтернативный сервис был использован используется вместе с Alt-Svc

## Authorization

предоставляет реквизиты для аутентификации на сервере

```bash
Authorization: <auth-scheme> <authorization-parameters>

# auth-scheme -  Basic, Digest, Negotiate,  AWS4-HMAC-SHA256.

# Basic authentication
Authorization: Basic <credentials>

# Digest authentication
Authorization: Digest username=<username>,
    realm="<realm>",
    uri="<url>",
    algorithm=<algorithm>,
    nonce="<nonce>",
    nc=<nc>,
    cnonce="<cnonce>",
    qop=<qop>,
    response="<response>", # hex
    opaque="<opaque>"
```

## Available-Dictionary

Это ID словаря, а не хеш

# Cookie

содержит хранимые cookie, которые были установлены Set-Cookie или js

```bash
Cookie: PHPSESSID=298zf09hf012fh2; csrftoken=u32t4o3tb3gg43; _gat=1
```

# Expect

Указывает ожидание, которое должен выполнить сервер. Единственно значение

```bash
Expect: 100-continue

# пример
PUT /somewhere/fun HTTP/1.1
Host: origin.example.com
Content-Type: video/h264
Content-Length: 1234567890987 # собираемся с клиента загрузить большой файл
Expect: 100-continue
```

Может быть 417
