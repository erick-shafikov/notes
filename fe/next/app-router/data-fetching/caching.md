# Кеширование

типы кеширования:

# мемоизация запросов

next оборачивает стандартный fetch, который в свю очередь помогает react на стороне сервера мемоизировать данные для React на стороне клиента

- можно вызывать в разных компонентах
- если не используется fetch, то нужно использовать cache из react
- применяется только к get

- Продолжительность:
- - время срока запроса к серверу, до завершения рендеринга дерева компонентов React
- сброс:
- - AbortController в fetch

# кеш данных

кеширование данных запросов на стороне сервера

- Продолжительность
- - кешируется все, до момента отключения или ревалидации

```js
//по умолчанию
fetch("https://...", { cache: "force-cache" });
//такой запрос будет обращен к источнику
fetch("https://...", { cache: "no-store" });
```

## ревалидирование данных

- ревалидация данных происходит в фоновом режиме

## time-based

```js
fetch("https://...", { next: { revalidate: 3600 } });
```

или

```js
export const revalidate = 3600; //из layout или page файла
```

## on demand

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

## отключение

отменить можно с помощью cache next.revalidate, при cache: 'no-store' запросы будут идти в источник

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

# полный кеш маршрута

автоматическая статическая оптимизация, статическая генерация сайта, статический рендеринг - понятия для кеширования маршрутов

- автоматически кешируется во время сборки все маршруты
- использует рендеринг react на сервере вместе с rscp и html

# кеш на стороне клиента

React Server Component Payload хранится в клиентском кеше, используется для навигации, при навигации проверит, хранится ли что-либо в кеше роутера. Кеширование роута во время сборки зависит статический ли он или динамический

- сброс:
- - при повторной сборке
- - при повторной проверке данных
- отключение:
- - dynamic = 'force-dynamic' или revalidate = 0,
- - отключение кеширования в fetch
- продолжительность:
- - сеанс
- - prefetch={null}
- аннулирование:
- - revalidatePath, revalidateTag
- - использование cookies.set или cookies.delete
- - router.refresh

# взаимосвязь

Кэш данных и полный кэш маршрутов:

- проверка в кеше данных привет к аннулированию в кеше маршрута
- кеш роутера не влияет на кеш данных

Кэш данных и кэш клиентского маршрутизатора

- сброс обоих с помощью revalidatePath или revalidateTag в server action

# кеш в разных api

## Link

```tsx
//отключит кеширование
<Link prefetch={false} />
```

# useRouter

- router.prefetch() вызывает предварительное кеширование
- router.refresh() очищает кэш маршрутизатора, не влияет на кеш данных
- fetch - по умолчанию не кешируется в кеше данных
- - fetch(`https://...`, { cache: 'force-cache' }) - активировать
- - fetch(`https://...`, { next: { revalidate: 3600 } }) по времени
- - fetch(`https://...`, { next: { tags: ['a', 'b', 'c'] } }) по требованию
- - или revalidateTag('a'), revalidatePath('/') использовать в route handlers и revalidatePath

# конфигурация сегмента

если нельзя сконфигурировать в fetch

исключение:

const dynamic = 'force-dynamic'
const fetchCache = 'default-no-store'

generateStaticParams кэшируются в кэше полного маршрута во время сборки

cache Функция react
