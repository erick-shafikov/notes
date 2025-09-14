# createFileRoute

за создание корневых роутов отвечают:

- [createRootRoute](./functions/createRootRoute.md)
- [createRootRouter](./functions/createRootRouter.md)

Компоненты роутинг создаются с помощью:

- [createFileRoute](./functions/createFileRoute.md)

# маршруты

Основные типы:

- перманентные - /post
- динамические - /post/$id

Доступ к параметрам

```tsx
import { createFileRoute, Link, Outlet } from "@tanstack/react-router";
import type { Post } from "../../types/posts";

// типизация для фильтров
type ProductSearch = {
  page: number;
};

// '/posts/{-$postId}' - если postId необязательный параметр
// posts.{-$category}.{-$slug}.tsx - вложенный пример
// export const Route = createFileRoute('/posts/{-$category}/{-$slug}')({
//   component: PostsComponent,
// })
export const Route = createFileRoute("/posts/$postId")({
  loader: ({ params }) => {
    // будет доступно по params.postId
  },
  component: PostComponent,
  // валидация для фильтров + типизация
  validateSearch: (search: Record<string, unknown>): ProductSearch => {
    return {
      page: Number(search?.page ?? 1),
    };
  },
});

function PostComponent() {
  const { page } = Route.useSearch();
  const { postId } = Route.useParams();

  return <></>;
}
```

# маршруты layout

вариант 1

routes/
├─app.tsx - здесь должен быть outlet компонент
├─app.dashboard.tsx
├─app.settings.tsx

вариант 2

routes/
├─ app/
│ ├─route.tsx - здесь должен быть outlet компонент
│ ├─dashboard.tsx
│ ├─settings.tsx

# маршруты \_layout

отобразится лишь только в том случае если перейдем на \_pathlessLayout.a или \_pathlessLayout.b

routes/
├─_pathlessLayout.tsx
├─_pathlessLayout.a.tsx
├─_pathlessLayout.b.tsx

если с директорией route

routes/
├─_pathlessLayout/
│ ├─route.tsx
│ ├─a.tsx
│ ├─b.tsx

если вынести определенный файл posts\_ из layout

routes/
├─posts.tsx
├─posts.$postId.tsx
├─posts_.$postId.edit.tsx

# исключения из маршрутизации

routes/
├─posts.tsx
├─-posts-table.tsx // 👈🏼 ignored
├─-components/ // 👈🏼 ignored
│ ├─header.tsx // 👈🏼 ignored
│ ├─footer.tsx // 👈🏼 ignored

# группировка

routes/
├─index.tsx
├─(app)/
│ ├─dashboard.tsx
│ ├─settings.tsx
│ ├─users.tsx
├─(auth)/
│ ├─login.tsx
│ ├─register.tsx

- пример

/routes
├─\_\_root.tsx
├─index.tsx
├─about.tsx
├─posts/
│ ├─index.tsx
│ ├─$postId.tsx
├─posts.$postId.edit.tsx
├─settings/
│ ├─profile.tsx
│ ├─notifications.tsx
├─_pathlessLayout/
│ ├─route-a.tsx
├─├─route-b.tsx
├─files/
│ ├─$.tsx

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
