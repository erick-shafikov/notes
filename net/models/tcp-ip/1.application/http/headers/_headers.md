# заголовки

Делятся по контексту:

- те которые относятся к запросам и ответам
- заголовки запроса
- заголовки ответа
- заголовки репрезентативные - предоставляют информацию о работе с телом запроса
- заголовки тела запроса - данные о теле запроса
- Группировка по прокси:
- - end-to-end - не влияющие на прокси
- - hop-by-hop - заголовки для транспортного уровня задаваемые заголовком [Connection](./deprecated.md#connection-req-res) не должны повторно передаваться прокси-серверами или кэшироваться.

Некоторые заголовки поддерживают степень поддержки того или иного значения:

```bash
Accept-Encoding: br;q=1.0, gzip;q=0.8, \;q=0.1
```

Типы заголовков по функционалу

accept заголовки - для согласования контента:

- [accept-language](./req-headers.md#accept-language)
- [accept-ch](./res-headers.md#accept-ch)
- [accept](./req-res-headers.md#accept)
- [accept-encoding](./req-res-headers.md#accept-encoding)
- [accept-ranges](./req-res-headers.md#accept-ranges)
- [accept-patch](./res-headers.md#accept-patch)
- [accept-post](./res-headers.md#accept-post)
- [preflight заголовки](./preflight-headers.md)

для работы с прокси и кешированием:

- [age](./res-headers.md#age)
- [cache-control](./req-res-headers.md#cache-control)
- [expires](./res-headers.md#expires)
- [no-vary-search](./experemental.md#no-vary-search-res)
- [pragma](./deprecated.md#pragma-req-res)
- [Referer](./req-headers.md#referer)
- [vary](./res-headers.md#vary)
- [via](./req-res-headers.md#via)
- [x-forwarded-for](./x-headers.md#x-forwarded-for-req)

для работы со словарями

- [Available-Dictionary](./req-headers.md#available-dictionary)
- [dictionary-id](./experemental.md#dictionary-id-req-res--ff--sf)
- [use-as-dictionary](./experemental.md#use-as-dictionary)

для работы с контентом:

- [representation заголовки](./representation-headers.md)
- [content-disposition](./req-res-headers.md#content-disposition)
- [etag](./res-headers.md#etag)
- [expect](./req-headers.md#expect)
- [idempotency-key](./experemental.md#idempotency-key-req)
- [range](./req-headers.md#range)
- [te](./req-headers.md#te)
- [x-content-type-options](./x-headers.md#x-content-type-options-res)

для управление CSP и безопасностью:

- [content-security-policy](./res-headers.md#content-security-policy)
- [content-security-policy-report-only](./res-headers.md#content-security-policy-report-only)
- [cross-origin-embedder-policy](./res-headers.md#cross-origin-embedder-policy)
- [cross-origin-opener-policy](./res-headers.md#cross-origin-opener-policy)
- [cross-origin-resource-policy](./res-headers.md#cross-origin-resource-policy)
- [integrity-policy](./res-headers.md#integrity-policy)
- [permissions-policy](./experemental.md#permissions-policy-res)
- [referrer-policy](./res-headers.md#referrer-policy)
- [reporting-endpoints](./res-headers.md#reporting-endpoints)
- [security-заголовки](./sec-headers.md)
- [x-frame-options](./x-headers.md#x-frame-options-res)
- [x-xss-protection](./x-headers.md#x-xss-protection-res-dep)

информация о запросе и ответе:

- [date](./req-res-headers.md#date)
- [forwarded](./req-headers.md#forwarded)
- [host](./req-headers.md#host)
- [nel](./experemental.md#nel-res)
- [origin](./req-headers.md#origin)
- [prefer](./req-headers.md#prefer)
- [preference-applied](./res-headers.md#preference-applied)
- [priority](./req-res-headers.md#priority)
- [repr-digest](./req-res-headers.md#repr-digest)
- [server](./res-headers.md#server)
- [server-timing](./res-headers.md#server-timing)
- [user-agent](./req-headers.md#user-agent)
- [x-powered-by](./x-headers.md#x-powered-by-res)

информация о клиенте (сh):

- [critical-ch](./experemental.md#critical-ch-res)
- [dnt-req](./deprecated.md#dnt-req)
- [downlink](./experemental.md#downlink-req)
- [content-dpr](./deprecated.md#content-dpr-req)
- [dpr-req](./deprecated.md#dpr-req)
- [ect-req](./experemental.md#ect-req)
- [origin-agent-cluster](./res-headers.md#origin-agent-cluster)
- [rtt](./experemental.md#rtt-req)
- [save-data](./experemental.md#save-data-req)

для роботов:

- [from](./req-headers.md#from)
- [x-robots-tag](./x-headers.md#x-robots-tag-res)

conditional:

- [if-match](./req-headers.md#if-match)
- [if-modified-since](./req-headers.md#if-modified-since)
- [if-none-match](./req-headers.md#if-none-match)
- [if-range](./req-headers.md#if-range)
- [if-unmodified-since](./req-headers.md#if-unmodified-since)
- [last-modified](./res-headers.md#last-modified)

hop-by-hop - для управление соединением:

- [connection](./deprecated.md#connection-req-res)
- [keep-alive](./deprecated.md#keep-alive)
- [transfer-encoding](./req-res-headers.md#transfer-encoding)

Методы аутентификации:

- [proxy-authenticate](./res-headers.md#proxy-authenticate)
- [proxy-authorization](./req-headers.md#proxy-authorization)
- [set-cookie](./res-headers.md#set-cookie)
- [cookie](./req-headers.md#cookie)
- [www-authenticate](./res-headers.md#www-authenticate)
