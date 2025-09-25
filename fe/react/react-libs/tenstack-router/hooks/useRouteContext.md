# useRouteContext

для доступа к контексту из компонентов

Параметры:

- Принимает (объект):
- - from
- - select
- Возвращает:
- - контекст

```tsx
import { useRouteContext } from "@tanstack/react-router";

function Component() {
  const context = useRouteContext({ from: "/posts/$postId" });

  const selected = useRouteContext({
    from: "/posts/$postId",
    select: (context) => context.postId,
  });

  // ...
}
```
