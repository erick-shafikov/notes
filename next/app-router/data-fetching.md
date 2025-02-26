CC – client component
SC – server component

## fetching, caching, revalidating

Варианты загрузки данных:

1. На сервере с помощью fetch (в route-handlers, серверных компонентах, серверных экшенах)
2. На сервере с мощью сторонних библиотек
3. На клиенте с помощью Rout-Handler
4. На клиенте с мощью сторонних библиотек

Кэширование

2 варианта ревалидирования данных - **time-based** и **on-demand**

TIME-BASED

- По умолчанию кэширование включено: fetch('https://...', { cache: 'force-cache' }) или export const revalidate = 3600
- Для time-based кеширования fetch('https://...', { next: { revalidate: 3600 } }) или export const revalidate = 3600 из layout или page файла

ON DEMAND

```js
// В fetch
const res = await fetch("https://...", { next: { tags: ["collection"] } });

// или в серверном action

("use server");

import { revalidateTag } from "next/cache";

export default async function action() {
  revalidateTag("collection");
}
```

**НЕ КЕШИРУЕТСЯ**

```js
fetch("https://...", { cache: "no-store" });
fetch("https://...", { next: { revalidate: 0 } });
```

- В POST методе в Route Handlers
- При применении cookies и headers функций
- const dynamic = 'force-dynamic'
- При наличие авторизационных заголовков
- С использованием cache (хук из react)
