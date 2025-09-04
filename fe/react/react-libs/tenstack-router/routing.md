# createFileRoute

с помощью этой функции создаются и регистрируются роуты, принимает один параметр - путь

```tsx
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: PostsComponent,
});
```

# createRootRoute

```tsx
// без контекста
import { createRootRoute } from "@tanstack/react-router";
import { createRootRouteWithContext } from "@tanstack/react-router";

export const Route = createRootRoute();

//с контекстом
export interface MyRouterContext {
  //типизация
}

// компонент созданный createRootRoute всегда отображается
// доступны компоненты, загрузка, проверка параметров
export const Route = createRootRouteWithContext<MyRouterContext>()({
  component: () => <>...</>,
  // ...
});
```

```tsx
//в main.tsx
const router = createRouter({
  routeTree,
  context: {
    //контекст для роутинга
   },
});

// регистрация типов (по умолчанию)
declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const rootElement = document.getElementById("root")!;
if (!rootElement.innerHTML) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </StrictMode>,
  );
```

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

№ пример

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
