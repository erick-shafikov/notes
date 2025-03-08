## composition-patterns

- вместо контекста можно воспользоваться кешированием fetch функции
- использование server-only (библиотека)
- Использование сторонних библиотек через ре-экспорт в клиентских компонентах
- использовать клиентские компоненты ниже в дереве компонентов
- Серверные компоненты нельзя экспортировать в клиентские, но их можно передать пропсом (как вариант в children)

```jsx
const ContextProvider = ({ children }) => {
  const [s, ss] = useState();
  return <Context.Provider value={s}>{children}</Context.Provider>;
};
```

Приоритетность рендеринга:
static - нет динамики, закешированные данные, не смотрим в кукуи (generateStaticParams для динамики)
dynamic - если не сработал static

пакет 'server-only' позволяет сообщать при разработке что серверные компоненты вызываются на клиенте

```ts
import "server-only";
//при импорте в клиентские компоненты получим ошибку
export async function getData() {
  const res = await fetch("https://external-service.com/data", {
    headers: {
      authorization: process.env.API_KEY,
    },
  });

  return res.json();
}
```

# использование библиотек

Можно использовать ре-экспорт для использования в клиентских компонентах

```tsx
//определяем библиотечный компонент, как клиентский
"use client";

import { Carousel } from "acme-carousel";

export default Carousel;
```

используем на сервере

```tsx
import Carousel from "./carousel";

export default function Page() {
  return (
    <div>
      <p>View pictures</p>

      {/*  Works, since Carousel is a Client Component */}
      <Carousel />
    </div>
  );
}
```
