# createRootRoute

регистрирует корневой роут

Принимает:

- TypeRootRoute [RootOPtions](../types/RouteOptions.md)

```ts
type TypeRootRoute = Omit<
  RouteOptions,
  | "path"
  | "id"
  | "getParentRoute"
  | "caseSensitive"
  | "parseParams"
  | "stringifyParams"
>;
```

```tsx
// без контекста
import { createRootRoute, createRouter, Outlet } from "@tanstack/react-router";

const rootRoute = createRootRoute({
  component: () => <Outlet />,
  // ... root route options
});

const routeTree = rootRoute.addChildren([
  // ... other routes
]);

const router = createRouter({
  routeTree,
});
```
