# createFileRoute

за создание корневых роутов отвечают:

- [createRootRoute - создаст корневой компонент роутинга, создается в \_\_root](./functions/createRootRoute.md)
- [createRouter - создаст конфигурацию роутинга вызывается в main и передается в провайдер](./functions/createRouter.md)

Компоненты роутинг создаются с помощью:

- [createFileRoute для файлового роутинга](./functions/createFileRoute.md)
- [createRoute для программного роутинга](./functions/createRoute.md)

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

# file-based подход

## маршруты layout

вариант 1

routes/
├─app.tsx ⇒ AppLayout здесь должен быть outlet компонент
├─app.dashboard.tsx ⇒ AppLayout[Dashboard]
├─app.settings.tsx ⇒ AppLayout[Settings]

вариант 2

routes/
├─ app/
│ ├─route.tsx - здесь должен быть outlet компонент, это файл конфигурации
│ ├─dashboard.tsx
│ ├─settings.tsx

# маршруты \_layout

отобразится лишь только в том случае если перейдем на \_pathlessLayout.a или \_pathlessLayout.b. \_pathlessLayout - будет оберткой. Если есть route будет внутри него

routes/
├─_pathlessLayout.tsx ⇒ index
├─_pathlessLayout.a.tsx ⇒ PathlessLayout[A]
├─_pathlessLayout.b.tsx ⇒ PathlessLayout[B]

- !!! нельзя \_$postId/
- ├── $postId/ можно
  ├── \_postPathlessLayout/

если с директорией route

routes/
├─_pathlessLayout/
│ ├─route.tsx
│ ├─a.tsx
│ ├─b.tsx

если вынести определенный файл posts\_ из layout

routes/
├─posts.tsx ⇒ Posts
├─posts.$postId.tsx  ⇒ Posts[Post postId="123"]
├─posts_.$postId.edit.tsx ⇒ PostEditor postId="123" вне Posts

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

\_\_root.tsx ⇒ Root
index.tsx ⇒ exact Root[RootIndex] (/)
about.tsx ⇒ Root[About] (/about)
posts.tsx ⇒ Root[Posts] (/posts)

# дерево

📂 posts:

- index.tsx ⇒ exact Root[Posts[PostsIndex]] (/posts)
- $postId.tsx ⇒ Root[Posts[Post]] (/posts/$postId)

📂 posts\_:

- 📂 $postId:
- - edit.tsx ⇒ Root[EditPost] (/posts/$postId/edit)

settings.tsx ⇒ Root[Settings] /settings
📂 settings Root[Settings] :

- profile.tsx ⇒ Root[Settings[Profile]] (/settings/profile)
- notifications.tsx ⇒ Root[Settings[Notifications]] (/settings/notifications)

\_pathlessLayout.tsx ⇒Root[PathlessLayout]
📂 \_pathlessLayout:

- route-a.tsx ⇒ Root[PathlessLayout[RouteA]] (/route-a)
- route-b.tsx ⇒ Root[PathlessLayout[RouteB]] (/route-b)

📂 files:

- $.tsx ⇒ Root[Files] (/files/$)

📂 account:

- route.tsx ⇒ Root[Account] (/account)
- overview.tsx ⇒ Root[Account[Overview]] (/account/overview)

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

# .lazy-файлы

```tsx
// src/routes/posts.tsx

import { createFileRoute } from "@tanstack/react-router";
import { fetchPosts } from "./api";

export const Route = createFileRoute("/posts")({
  loader: fetchPosts,
});
```

```tsx
// src/routes/posts.lazy.tsx

import { createLazyFileRoute } from "@tanstack/react-router";

export const Route = createLazyFileRoute("/posts")({
  component: Posts,
});

function Posts() {
  // ...
}
```
