# useLoaderData

Параметры:

- Принимает (объект с полями):
- - from - строка, опциональная, но рекомендуется
- - strict - true, если false то from игнорируется
- - select - (loaderData: TLoaderData) => TSelected
- - structuralSharing - boolean для select, для оптимизации
- Возвращает
- - данные либо после преобразования в select либо просто данные

```tsx
import { useLoaderData } from "@tanstack/react-router";

function Component() {
  const loaderData = useLoaderData({ from: "/posts/$postId" });
  //     ^? { postId: string, body: string, ... }
  // ...
}
```
