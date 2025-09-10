# Route

```tsx
import {
  createRouter,
  parseSearchWith,
  stringifySearchWith,
} from "@tanstack/react-router";

export const Route = createFileRoute("/posts/$postId")({
  component: PostComponent,
  // пред-загрузка данных
  defaultPreload: "intent",
  // управление head компонентом [1]
  head: () => ({}),
  // управление парсингом поисковой строки
  parseSearch: parseSearchWith(JSON.parse),
  stringifySearch: stringifySearchWith(JSON.stringify),
  //управление временем лоудеров
  preloadStaleTime: 10_000,
  defaultPreloadStaleTime: 10_000,
});
```

# управление head

если нужно управлять заголовками в документ нужно подключить HeadContent

```tsx
//page.tsx
export const Route = createFileRoute("/posts/$postId")({
  //...
  // переопределяем в заголовке данные
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
});

//корневой компонент
//что бы заработали заголовки нужно в корневой компонент добавить HeadContent
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
