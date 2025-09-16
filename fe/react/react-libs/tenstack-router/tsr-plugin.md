# code splitting

- автоматически в vite.config.ts autoCodeSplitting: true,
- использование lazy файлов

```tsx
// src/routes/posts.tsx
// файл с loader, в последствии можно без него если ненужен loader
import { createFileRoute } from "@tanstack/react-router";
import { fetchPosts } from "./api";

export const Route = createFileRoute("/posts")({
  loader: fetchPosts,
});

//lazy компонент в отдельном .lazy файле
// src/routes/posts.lazy.tsx
import { createLazyFileRoute } from "@tanstack/react-router";

export const Route = createLazyFileRoute("/posts")({
  component: Posts,
});

function Posts() {
  // ...
}
```

разделение на основе кода

```tsx
// src/posts.lazy.tsx
export const Route = createLazyRoute("/posts")({
  component: MyComponent,
});

function MyComponent() {
  return <div>My Component</div>;
}

// src/app.tsx
const postsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/posts",
}).lazy(() => import("./posts.lazy").then((d) => d.Route));
```

файл загрузки

```tsx
import { lazyFn } from "@tanstack/react-router";

const route = createRoute({
  path: "/my-route",
  component: MyComponent,
  loader: lazyFn(() => import("./loader"), "loader"),
});

// In another file...a
export const loader = async (context: LoaderContext) => {
  /// ...
};
```

Вынос логики

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

//my-component.tsx
import { getRouteApi } from "@tanstack/react-router";

const route = getRouteApi("/my-route");
// доступны
// useLoaderData
// useLoaderDeps
// useMatch
// useParams
// useRouteContext
// useSearch

export function MyComponent() {
  const loaderData = route.useLoaderData();
  //    ^? { foo: string }

  return <div>...</div>;
}
```
