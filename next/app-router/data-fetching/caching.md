# Кеширование

Два типа кеширования:
кеширование данных - при всех запросах
мемоизация запроса только при сроке действия запроса, для generateStaticParma то есть для дерева react

Плюсы:

- можно вызывать в разных компонентах

отменить можно с помощью cache next.revalidate, при cache: 'no-store' запросы будут идти в источник

2 варианта ревалидирования данных - time-based и on-demand

## TIME-BASED

- По умолчанию кэширование включено:

```js
fetch("https://...", { cache: "force-cache" });
```

- Для time-based кеширования

```js
fetch("https://...", { next: { revalidate: 3600 } });
```

или

```js
export const revalidate = 3600; //из layout или page файла
```

## ON DEMAND

```tsx
// В fetch
const res = await fetch("https://...", { next: { tags: ["collection"] } });
```

```tsx
// или в серверном action
"use server";

import { revalidateTag } from "next/cache";

export default async function action() {
  revalidateTag("collection");
}
```

# не кешируется

```js
//по умолчанию
fetch("https://...", { cache: "no-store" });
fetch("https://...", { next: { revalidate: 0 } });
```

```js
// установка тегов
fetch(`https://...`, { next: { tags: ["a", "b", "c"] } });

// ревалидирование
revalidateTag("a");
```

```tsx
<Link prefetch={false} />
```

- В POST методе в Route Handlers
- При применении cookies и headers функций
- const dynamic = 'force-dynamic'
- При наличие авторизационных заголовков
- С использованием cache (хук из react)

Поведение Next.js по умолчанию — кэшировать отрендеренный результат (React Server Component Payload и HTML) маршрута на сервер

Кэш маршрутизатора на стороне клиента
