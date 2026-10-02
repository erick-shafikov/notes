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
├─app.tsx ⇒ /app (layout, нужен Outlet)
├─app.dashboard.tsx ⇒ /app/dashboard
├─app.settings.tsx ⇒ /app/settings

вариант 2

routes/
├─ app/
│ ├─route.tsx ⇒ /app (layout, нужен Outlet)
│ ├─dashboard.tsx ⇒ /app/dashboard
│ ├─settings.tsx ⇒ /app/settings

вариант 3

routes/
├─app.tsx ⇒ /app (layout, нужен Outlet)
├─ app/
│ ├─dashboard.tsx ⇒ /app/dashboard
│ ├─settings.tsx ⇒ /app/settings

# маршруты \_layout

отобразится лишь только в том случае если перейдем на \_pathlessLayout.a или \_pathlessLayout.b. \_pathlessLayout - будет оберткой. Если есть route будет внутри него

routes/
├─_pathlessLayout.tsx ⇒ (нет URL, только обёртка — нужен Outlet)
├─_pathlessLayout.a.tsx ⇒ /a
├─_pathlessLayout.b.tsx ⇒ /b

- !!! нельзя \_$postId/
- ├── $postId/ можно
  ├── \_postPathlessLayout/

если с директорией route

routes/
├─_pathlessLayout/
│ ├─route.tsx ⇒ (нет URL, только обёртка — нужен Outlet)
│ ├─a.tsx ⇒ /a
│ ├─b.tsx ⇒ /b

если вынести определенный файл posts\_ из layout

routes/
├─posts.tsx ⇒ /posts (layout)
├─posts.$postId.tsx           ⇒ /posts/$postId (внутри layout posts.tsx)
├─posts\_.$postId.edit.tsx     ⇒ /posts/$postId/edit (вне layout posts.tsx)

# исключения из маршрутизации

routes/
├─posts.tsx ⇒ /posts
├─-posts-table.tsx ⇒ ignored (нет маршрута)
├─-components/ ⇒ ignored
│ ├─header.tsx ⇒ ignored
│ ├─footer.tsx ⇒ ignored

# группировка

routes/
├─index.tsx ⇒ /
├─(app)/ ⇒ (группировка, нет URL-сегмента)
│ ├─dashboard.tsx ⇒ /dashboard
│ ├─settings.tsx ⇒ /settings
│ ├─users.tsx ⇒ /users
├─(auth)/ ⇒ (группировка, нет URL-сегмента)
│ ├─login.tsx ⇒ /login
│ ├─register.tsx ⇒ /register

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

# index.tsx — точное совпадение

`index.tsx` внутри директории или рядом с layout-файлом — это маршрут `/` относительно родителя (exact match, без trailing slash):

routes/
├─ posts.tsx ⇒ /posts (layout)
├─ posts/
│ ├─ index.tsx ⇒ /posts (exact, рендерится внутри posts.tsx)
│ ├─ $postId.tsx ⇒ /posts/$postId

То есть `/posts` рендерит `posts.tsx` → `posts/index.tsx`, а `/posts/123` рендерит `posts.tsx` → `posts/$postId.tsx`.

# splat-маршруты

`$.tsx` — ловит любой подпуть, которому не нашлось совпадения. Параметр доступен через `params['*']`:

routes/
├─ files/
│ ├─ $.tsx ⇒ /files/anything/nested/here

```tsx
export const Route = createFileRoute("/files/$")({
  component: FilesComponent,
});

function FilesComponent() {
  const { "*": splat } = Route.useParams();
  // /files/a/b/c → splat === 'a/b/c'
  return <div>{splat}</div>;
}
```

Используется для file-browser'ов, catch-all страниц, проксирования подпутей.

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
