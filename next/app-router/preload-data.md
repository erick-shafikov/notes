Preload data

Для избежания задержки в отображении контента можно использовать preload

```js
import { getItem } from "@/utils/get-item";

export const preload = (id: string) => {
  // void выполняет функцию спарава и возвращает undefined
  // https://developer.mozilla.org/docs/Web/JavaScript/Reference/Operators/void
  void getItem(id);
};

export default async function Item({ id }: { id: string }) {
    const result = await getItem(id)
    // ... сам компонент
  }

  import Item, { preload, checkIsAvailable } from '@/components/Item'

  export default async function Page({ params: { id },}: { params: { id: string }}) {
    // starting loading item data
    preload(id) // используем предзагрузку
    const isAvailable = await checkIsAvailable()

    return isAvailable ? <Item id={id} /> : null
  }
```

Использование cache, server-only, Preload

данный подход гарантирует запрос к внешним источникам только на сервере

```js
import { cache } from "react";
import "server-only"; //дополнительная библиотека

export const preload = (id: string) => {
  void getItem(id);
};

export const getItem = cache(async (id: string) => {
  // ...
});
```
