# getRouteApi

позволяет получить экземпляр Route на любой из страниц

Параметры:

- принимает - строку
- [возвращает](../types/RouteApi.md)

```tsx
//my-route.tsx
import { createRoute } from "@tanstack/react-router";
import { MyComponent } from "./MyComponent";

const route = createRoute({
  path: "/my-route",
  loader: () => ({
    foo: "bar",
  }),
  component: MyComponent,
});
```

```tsx
// page.tsx
import { getRouteApi } from "@tanstack/react-router"; // доступ к данным
const routeApi = getRouteApi("/posts");

function PostComponent() {
  const {} = Route.useLoaderData();

  const data = routeApi.useLoaderData();

  return <></>;
}
```

Доступны:

- useLoaderData
- useLoaderDeps
- useMatch
- useParams
- useRouteContext
- useSearch
