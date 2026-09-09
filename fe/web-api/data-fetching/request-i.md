# Request

Объект запроса

# Свойства

## body (ro)

=> ReadableStream или null (для GET или HEAD).

## bodyUsed (ro)

=> boolean - прочитан ли стрим

## cache (ro)

=> default | no-store | reload | no-cache | force-cache | only-if-cached - о браузерном кеше запроса

## credentials (ro)

=> omit | same-origin | include

## destination (ro)

=> audio | audioworklet | document | embed | fencedframe | font | frame | iframe | image | json | manifest | object | paintworklet | report | script | sharedworker | speculationrules | style | text | track | video | worker or xslt

## duplex (ro -sf, -ff)

=> half

## headers

=> заголовки запроса

```ts
const myHeaders = new Headers();
myHeaders.append("Content-Type", "image/jpeg");

const myInit = {
  method: "GET",
  headers: myHeaders,
  mode: "cors",
  cache: "default",
};

const myRequest = new Request("flowers.jpg", myInit);

const myContentType = myRequest.headers.get("Content-Type"); // 'image/jpeg'
```

## integrity (ro)

что было передано в поле integrity при составлении запроса

## isHistoryNavigation (ro)

=> boolean - является ли браузерной навигацией

## isReloadNavigation (ro)

=> boolean - является ли результатом перезагрузки

## keepalive (ro)

=> boolean - передан ли был keepalive при конструировании запроса

## method (ro)

=> метод запроса

## mode (ro)

=> same-origin | no-cors | cors | navigate - тип запроса

## redirect (ro)

=> follow | error | manual - как был обработан редирект

## referrer (ro)

=> строка с referrer

## referrerPolicy (ro)

=> строка с referrerPolicy

## signal (ro)

=> [AbortSignal](./abortcontroller.md)

## targetAddressSpace (ro)

=> local | loopback | public | unknown

## url

=> строка

# Методы

## arrayBuffer()

=> Promise<ArrayBuffer> [ArrayBuffer](../array-buffers/ArraBuffer-i.md) побитовое представление запроса

## blob()

=> бинарное файловое представление [Blob](../files/blob-i.md)

## bytes()

=> Promise<Uint8Array> запроса

## clone()

=> копия запроса

## formData()

=> Promise<FormData>

## json()

=> Promise<Object>

## text()

=> Promise<string> utf-8 строка запроса

## textStream()

=> [ReadableStream](../array-buffers/readable-stream-i.md) - чанки utf8 []
