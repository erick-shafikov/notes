# репрезентативные и контентные заголовки

заголовки которые описывают как интерпретировать данные в сообщении

# Content-Encoding

тип сжатия body

```bash
Content-Encoding: gzip # LZ77
Content-Encoding: compress  # LZW
Content-Encoding: deflate  # zlib
Content-Encoding: identity # без сжатия
Content-Encoding: br # Brotli

# последовательность сжатий
Content-Encoding: gzip, identity
Content-Encoding: deflate, gzip
```

вместе с [Accept-Encoding](#accept-encoding) могут договариваться с сервером по поводу сжатия

# Content-Language

Сообщает о том к какой языковой группе относится контент

```bash
Content-Language: de-DE
Content-Language: en-US
Content-Language: de-DE, en-CA
```

Может быть указан в html Документе

```html
<html lang="de"></html>
<!-- /!\ Это плохая практика -->
<meta http-equiv="content-language" content="de" />
```

# Content-Length (res, req, cont)

размер в байтах. Передает информацию если идет стрим контента или генерация контента

# Content-Location

альтернативное расположение возвращаемых данных, в отличие от Location указывает прямую ссылку при [согласовании контента](../сontent-negotiation.md). Location предполагает 3ХХ код ответа для редиректа

```bash
# Запрос
Accept: application/json, text/json
# Ответ
Content-Location: /documents/foo.json
# Запрос
Accept: application/xml, text/xml
# Ответ
Content-Location: /documents/foo.xml
# Запрос
Accept: text/plain, text/*
# Ответ
Content-Location: /documents/foo.txt
```

Как пример - при создании выплаты в Content-Location может быть помещен путь где находится страница с результатом выплаты

# Content-Range

показывает где находится содержимое тело ответа относительно ресурса

```bash
HTTP/2 206
content-type: image/jpeg
content-length: 1024
content-range: bytes 0-1023/146515
…

(binary content)

# или при неопределенном размере
Content-Range: bytes */67589
```
