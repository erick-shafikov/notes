# заголовки

Делятся по контексту:

- те которые относятся к запросам и ответам
- заголовки запроса
- заголовки ответа
- заголовки репрезентативные - предоставляют информацию о работе с телом запроса
- заголовки тела запроса - данные о теле запроса
- Группировка по прокси:
- - end-to-end - не влияющие на прокси
- - hop-by-hop - заголовки для транспортного уровня задаваемые заголовком [Connection](./deprecated-experemental.md#connection-req-res) не должны повторно передаваться прокси-серверами или кэшироваться.

Некоторые заголовки поддерживают степень поддержки того или иного значения:

```bash
Accept-Encoding: br;q=1.0, gzip;q=0.8, \;q=0.1
```

Типы заголовков по функционалу

- accept заголовки - для согласования контента:
- - [accept-language](./req-headers.md#accept-language)
- - [accept-ch](./res-headers.md#accept-ch)
- - [accept](./req-res-headers.md#accept)
- - [accept-encoding](./req-res-headers.md#accept-encoding)
- - [accept-ranges](./req-res-headers.md#accept-ranges)
- - [accept-patch](./res-headers.md#accept-patch)
- - [accept-post](./res-headers.md#accept-post)
- [preflight заголовки](./preflight-headers.md)
- для работы с прокси:
- - [age](./res-headers.md#age)
- - [cache-control](./req-res-headers.md#cache-control)
- для работы со словарями
- - [Available-Dictionary](./req-headers.md#available-dictionary)
- для работы с контентом:
- - [representation заголовки](./representation-headers.md)
- - [content-digest](./req-res-headers.md#content-digest)
- - [content-disposition](./req-res-headers.md#content-disposition)
