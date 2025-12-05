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

--aBoundaryString
Content-Disposition: form-data; name="myFile"; filename="img.jpg"
Content-Type: image/jpeg

# (data)
--aBoundaryString
Content-Disposition: form-data; name="myField"

# (data)
--aBoundaryString
# (more subparts)
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

# запросы диапазона

- проверка сервера на поддержку с помощью Accept-Ranges заголовка

```bash
# флаг -I == head запрос
curl -I https://i.imgur.com/z4d4kWk.jpg
# создаст запрос
HEAD /z4d4kWk.jpg HTTP/2
Host: i.imgur.com
User-Agent: curl/8.7.1
Accept: */*
# ответ
HTTP/2 200
content-type: image/jpeg
last-modified: Thu, 02 Feb 2017 11:15:53 GMT
accept-ranges: bytes
content-length: 146515

# после - запрос части информации первые 1024 байта, команда:
curl https://i.imgur.com/z4d4kWk.jpg -i -H "Range: bytes=0-1023" --output -
# создаст запрос
GET /z4d4kWk.jpg HTTP/2
Host: i.imgur.com
User-Agent: curl/8.7.1
Accept: */*
Range: bytes=0-1023
# ответ:
HTTP/2 206
content-type: image/jpeg
content-length: 1024
content-range: bytes 0-1023/146515

(binary content)

```
