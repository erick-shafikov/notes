# patterns-and-bp

Streaming

Пользователю не нужно ждать загрузки всех данных, чтобы взаимодействовать со страницей
Нужно использовать layouts и параллельные роуты в паре со Suspense

Параллельное и последовательное запросы
Последовательное возникает, когда запрос за данными идет во вложенных компонентах.
Пример со список с id и переходом по отдельным статьям. Можно использовать Suspense для показа preloader'а

Параллельное можно запустить с помощью PromiseAll для запросов и обернув все в Suspense

Preload data

Для избежания задержки в отображении контента можно использолвать preload

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

# rendering

- Есть клиентская среда выполнения и серверная
- Request – response lifecycle

## composition-patterns

- вместо контекста можно воспользоваться кешированием
- использование server-only (библиотека)
- Использование сторонних библиотек через ре-экспорт в клиентских компонентах
- использовать клиентские компоненты ниже в дереве компонентов
- Серверные компоненты нельзя экспортировать в клиентские, но их можно передать пропсом

```jsx
const ContextProvider = ({ children }) => {
  const [s, ss] = useState();
  return <Context.Provider value={s}>{children}</Context.Provider>;
};
```

Приоритетность рендеринга:
static - нет динамики, закешированные данные, не смотрим в кукуи (generateStaticParams для динамики)
dynamic - если не сработал static