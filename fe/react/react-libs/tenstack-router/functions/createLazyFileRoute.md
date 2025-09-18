# createLazyFileRoute

возвращает Route у которого есть только

```ts
type LazyFileRouteParams = Pick<
  RouteOptions,
  "component" | "pendingComponent" | "errorComponent" | "notFoundComponent"
>;
```

```tsx
import { createLazyFileRoute } from "@tanstack/react-router";

export const Route = createLazyFileRoute("/")({
  component: IndexComponent,
});

function IndexComponent() {
  const data = Route.useLoaderData();
  return <div>{data}</div>;
}
```
