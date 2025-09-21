# createLazyRoute

используется для создания ленивого роута

Параметры:

- принимает один аргумент string - идентификатор маршрута
- возвращает Route который принимает частично RouteOptions

```ts
type T = Pick<
  RouteOptions,
  "component" | "pendingComponent" | "errorComponent" | "notFoundComponent"
>;
```

```tsx
// src/route-pages/index.tsx
import { createLazyRoute } from "@tanstack/react-router";

export const Route = createLazyRoute("/")({
  component: IndexComponent,
});

function IndexComponent() {
  const data = Route.useLoaderData();
  return <div>{data}</div>;
}

// src/routeTree.tsx
import {
  createRootRouteWithContext,
  createRoute,
  Outlet,
} from "@tanstack/react-router";

interface MyRouterContext {
  foo: string;
}

const rootRoute = createRootRouteWithContext<MyRouterContext>()({
  component: () => <Outlet />,
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
}).lazy(() => import("./route-pages/index").then((d) => d.Route));

export const routeTree = rootRoute.addChildren([indexRoute]);
```
