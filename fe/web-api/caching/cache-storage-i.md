# CacheStorage

предоставляет доступ к объектам [Cache](./cache-i.md) хранилище всех именованных кешей, к которым можно получить доступ из [ServiceWorker](../workers/service-sorker.md)

Доступно через глобальное свойство caches

# Методы

## delete()

```ts
// delete(cacheName)

this.addEventListener("activate", (event) => {
  const cachesToKeep = ["v2"];

  event.waitUntil(
    caches.keys().then((keyList) =>
      Promise.all(
        keyList.map((key) => {
          if (!cachesToKeep.includes(key)) {
            return caches.delete(key);
          }
          return undefined;
        }),
      ),
    ),
  );
});
```

## has()

```ts
// has(cacheName) => Promise<boolean>
```

```ts
caches
  .has("v1")
  .then((hasCache) => {
    if (!hasCache) {
      someCacheSetupFunction();
    } else {
      caches.open("v1").then((cache) => cache.addAll(myAssets));
    }
  })
  .catch(() => {
    // Handle exception here.
  });
```

## keys()

```ts
// keys() => Promise<string[]>
```

## match()

позволяет узнать соответствие с запросом

```ts
interface Options {
  ignoreSearch: boolean;
  ignoreMethod: boolean;
  ignoreVary: boolean;
  cacheName: String;
}

type match = (request: Request, options: Options) => Promise;

caches.match(request, options).then(function (response) {
  // Какие-либо действия с response
});
```

## open(cacheName)

Параметры:

- cacheName - имя кеша

=> Promise<Cache>
