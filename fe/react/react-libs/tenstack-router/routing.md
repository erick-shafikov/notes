# createFileRoute

за создание корневых роутов отвечают:

- [createRootRoute - создаст корневой компонент роутинга](./functions/createRootRoute.md)
- [createRootRouter - создаст конфигурацию роутинга](./functions/createRootRouter.md)

Компоненты роутинг создаются с помощью:

- [createFileRoute](./functions/createFileRoute.md)
- [createRoute](./functions/createRoute.md)

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

# 404

Режимы отображения 404:

- foozy-mode - ближайший маршрут с компонентом 404 (по умолчанию)
- root-mode - все будет обработано notFoundComponent корневым компонентом

Реализация:

- [notFoundComponent в createFileRoute](./functions/createFileRoute.md)
- [компонент по умолчанию в createRouter](./functions/createRouter.md)

Можно пробросить ошибку notFound с помощью [notFound](./functions/notFound.md)

# авторизация

Основной вариант Опция route.beforeLoad c помощью функции redirect

```tsx
export const Route = createFileRoute("/_authenticated")({
  beforeLoad: async ({ location }) => {
    if (!isAuthenticated()) {
      throw redirect({
        to: "/login",
        search: {
          // Use the current location to power a redirect after login
          // (Do not use `router.state.resolvedLocation` as it can
          // potentially lag behind the actual current location)
          redirect: location.href,
        },
      });
    }
  },
});
```

без перенаправления

```tsx
export const Route = createFileRoute("/_authenticated")({
  component: () => {
    if (!isAuthenticated()) {
      return <Login />;
    }

    return <Outlet />;
  },
});
```
