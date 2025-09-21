# lazyRouteComponent

создает компонент с разделением коду

Параметры:

- принимает:
- - importer () => Promise<T>
- - exportName
- возвращает:
- - React.lazy компонент который можно пред-загрузить component.preload()

```ts
import { lazyRouteComponent } from "@tanstack/react-router";

const route = createRoute({
  path: "/posts/$postId",
  component: lazyRouteComponent(() => import("./Post")), // default export
});

// или

const route = createRoute({
  path: "/posts/$postId",
  component: lazyRouteComponent(
    () => import("./Post"),
    "PostByIdPageComponent" // named export
  ),
});
```
