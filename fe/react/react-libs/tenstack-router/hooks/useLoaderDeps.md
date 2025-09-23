# useLoaderDeps

объект с зависимости, которые запускают loader функцию

Параметры:

- Принимает:
- - from - string! - путь
- - select - (deps: TLoaderDeps) => TSelected
- - structuralSharing - boolean?
- Возвращает:
- - объект

```tsx
import { useLoaderDeps } from "@tanstack/react-router";

const routeApi = getRouteApi("/posts/$postId");

function Component() {
  const deps = useLoaderDeps({ from: "/posts/$postId" });

  // OR

  const routeDeps = routeApi.useLoaderDeps();

  // OR

  const postId = useLoaderDeps({
    from: "/posts",
    select: (deps) => deps.view,
  });

  // ...
}
```
