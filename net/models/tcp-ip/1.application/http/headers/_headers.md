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
- - [content-disposition](./req-res-headers.md#content-disposition)
- - [etag](./res-headers.md#etag)
- для управление CSP:
- - [content-security-policy](./res-headers.md#content-security-policy)
- - [content-security-policy-report-only](./res-headers.md#content-security-policy-report-only)
- - [cross-origin-embedder-policy](./res-headers.md#cross-origin-embedder-policy)
- - [cross-origin-opener-policy](./res-headers.md#cross-origin-opener-policy)
- - [cross-origin-resource-policy](./res-headers.md#cross-origin-resource-policy)
- информация о запросе:
- - [date](./req-res-headers.md#date)
- информация о клиенте (сh)
- - [critical-ch](./deprecated-experemental.md#critical-ch-res)
- - [dnt-req](./deprecated-experemental.md#dnt-req)
- - [downlink](./deprecated-experemental.md#downlink-req)
- - [content-dpr](./deprecated-experemental.md#content-dpr-req)
- - [](./deprecated-experemental.md#dpr-req)
- - [](./deprecated-experemental.md#ect-req)
