# Скрипты

- Атрибут defer
- минификация
- разбиение длинных задач(более 50 мс)
- код на отдельные бандлы
- предварительная загрузка ключевых ресурсов

```html
<link rel="preload" as="script" href="script.js" />
<link rel="preload" as="style" href="style.css" />
```

- кеширование c помощью Service Workers

```js
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open("v1").then((cache) => {
      return cache.addAll(["/critical.css", "/main.js"]);
    })
  );
});
```

-tree shaking в скриптах (настройка в бандлере)
