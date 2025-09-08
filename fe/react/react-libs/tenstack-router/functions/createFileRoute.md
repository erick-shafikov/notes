# Route

```tsx
export const Route = createFileRoute("/posts/$postId")({
  component: PostComponent,
  // предзагрузка данных
  defaultPreload: "intent",
  // управление head компонентом [1]
  head: () => ({
    meta: [
      {
        name: "description",
        content: "My App is a web application",
      },
      {
        title: "My App",
      },
    ],
    links: [
      {
        rel: "icon",
        href: "/favicon.ico",
      },
    ],
    styles: [
      {
        media: "all and (max-width: 500px)",
        children: `p {
                  color: blue;
                  background-color: yellow;
                }`,
      },
    ],
  }),
  preloadStaleTime: 10_000,
  defaultPreloadStaleTime: 10_000,
});
```

# заголовки

если нужно управлять заголовками в документ нужно подключить HeadContent

```tsx
function PostComponent() {
  // получить параметры строки
  const { postId } = Route.useParams();
  return <div>Post {postId}</div>;
}

import { HeadContent } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: () => (
    <html>
      <head>
        <HeadContent />
      </head>
      <body>
        <Outlet />
      </body>
    </html>
  ),
});
```
