# createRoute

создает роут навигации. Настройки:

- [Параметры настройки](../types/RouteOptions.md)
- [Возвращает](../types/Route.md)

```tsx
import { createRoute } from "@tanstack/react-router";
import { rootRoute } from "./__root";

const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  loader: () => {
    return "Hello World";
  },
  component: IndexComponent,
});

function IndexComponent() {
  const data = Route.useLoaderData();
  return <div>{data}</div>;
}
```
