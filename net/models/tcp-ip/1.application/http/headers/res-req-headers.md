# заголовки запроса и ответа

# Accept-Encoding

сохранить одну копию, сжатую с помощью gzip, и другую — с помощью brotli. В паре Content-Encoding могут сервер может вернуть 406 Not Acceptable. Варианты по алгоритмам, могут использоваться с ;q=:

- - gzip - LZ77
- - compress - LZW
- - deflate - zlib + deflate
- - br - Brotli
- - zstd - Zstandard
- - dcb - Dictionary-Compressed Brotli
- - dcz - Dictionary-Compressed Zstandard
- - identity

# Accept-Ranges

поддерживаются ли range-запросы, для частей ресурса

# Cache-Control

определяет как будет работать кеш в браузере и промежуточных узлах - прокси и CDN. Основные понятия:

- cache
- Shared cache
- Private cache
- Store response
- Reuse response
- Revalidate response
- Fresh response
- Stale response
- Age

Значения делятся по следующим категориям

- Варианты управления кешированием:
- - public - может быть закеширован в любом месте
- - private - кеш для одного пользователя
- - no-cache - обязательный запрос на сервер при использования кешированных данных
- - only-if-cached - использование только закешированных данных
- Время жизни:
- - max-age=<seconds> - относительно от времени времени запроса
- - s-maxage=<seconds> - только для разделяемых кешей
- - min-fresh=<seconds> - запрос который актуален некоторое время
- - immutable - тело запроса не меняется
- Управление ре-валидацией и загрузкой:
- - must-revalidate - должен проверять
- - proxy-revalidate - только для прокси
- Другие:
- - no-store - не должно быть кеша
- - no-transform - не должны быть применимы преобразования

Значения по типу запроса:

- значения для запроса: max-age=<seconds>, max-stale[=<seconds>], min-fresh=<seconds>, no-cache, no-store, no-transform, only-if-cached
- значения для ответа:must-revalidate, no-cache, no-store, no-transform, public, private, proxy-revalidate, max-age=<seconds>, s-maxage=<seconds>

Инструкции:

- immutable
- stale-while-revalidate=<seconds>
- stale-if-error=<seconds>

соотношение запроса и ответа (req - res):

- max-age - max-age
- max-stale - X
- min-fresh - X
- X - s-maxage
- no-cache - no-cache
- no-store - no-store
- no-transform - no-transform
- only-if-cached - X
- X - must-revalidate
- X - proxy-revalidate
- X - must-understand
- X - private
- X - public
- X - immutable
- X - stale-while-revalidate
- stale-if-error - stale-if-error

```bash
# выключить кеш
Cache-Control: no-cache, no-store, must-revalidate
# статичный контент
Cache-Control: public, max-age=31536000
```

# Connection (dep)

использует только в http1, оставить ли соединение после запроса, значения keep-alive, close

# Content-Digest

алгоритм хеширования примененный к содержимому. Заголовок Want-Content-Digest запрашивает данные с хешированием, базируясь на Content-Encoding и Content-Range

```bash
# digest-algorithm - sha-512 and sha-256. Небезопасные - md5, sha (SHA-1), unixsum, unixcksum, adler (ADLER32) and crc32c.
# digest-value - захешированное значение
Content-Digest: <digest-algorithm>=<digest-value>

# Multiple digest algorithms
Content-Digest: <digest-algorithm>=<digest-value>,<digest-algorithm>=<digest-value>, …
```

```bash
# запрос с клиента
GET /items/123 HTTP/1.1
Host: example.com
Want-Content-Digest: sha-256=10, sha=
# ответ с сервера
HTTP/1.1 200 OK
Content-Type: application/json
Content-Digest: sha-256=:RK/0qy18MlBSVnWgjwz6lZEWjP/lF5HF9bvEF8FabDg=:

# {"hello": "world"}
```

# Content-Disposition

В случае ответа, то как будет отображаться в браузере ответ
В случае запроса, то как интерпретировать каждую часть бинарных данных

Директивы:

- name - атрибут поля формы
- filename - Content-Disposition: attachment будет интерпретироваться "сохранить как"
- filename\* - filename но только с кодированием

```bash
# заголовки ответа
Content-Disposition: inline
Content-Disposition: attachment
Content-Disposition: attachment; filename="filename.jpg"
# заголовки составного запроса
Content-Disposition: form-data
Content-Disposition: form-data; name="fieldName"
Content-Disposition: form-data; name="fieldName"; filename="filename.jpg"
```

Пример ниже заставит браузер сохранить страницу под именем cool.html

```bash
# Ответ, вызывающий диалог "Сохранить как":
200 OK
Content-Type: text/html; charset=utf-8
Content-Disposition: attachment; filename="cool.html"
Content-Length: 22

<HTML>Save me!</HTML>
```
