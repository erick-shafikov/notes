# createFileRoute

с помощью этой функции создаются и регистрируются роуты, принимает один параметр - путь

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
  // корневой notFoundComponent
  notFoundComponent: ({
    //есть доступ к данным
    data,
  }) => {
    return <p>This setting page doesn't exist!</p>;
  },
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

# функция loader

```tsx
export const Route = createFileRoute("/posts")(
  // интерфейс routeOptions
  {
    component: PostComponent,
    // интерфейс обработчика
    loader: ({
      abortController,
      cause, //'preload' | 'enter' | 'stay' - причина совпадения маршрута
      // в контекст можно передавать функции обработки данных
      context: { fetchPosts }, // использует контекст родителя и свой из beforeLoad [2]
      deps,
      path,
      location,
      parentMatchPromise,
      preload, //bool
      route,
    }) =>
      fetchPosts({
        // использование abortController
        signal: abortController.signal,
      }),
    //использование контекста с beforeLoad [2]
    beforeLoad: () => ({
      fetchPosts: () => console.info("foo"),
    }),
    loader: ({ context: { fetchPosts } }) => {
      console.info(fetchPosts()); // 'foo'
    },
    // Управление зависимостями для лоудера, для управления кешем
    loaderDeps: ({ search: { pageIndex, pageSize } }) => ({
      pageIndex,
      pageSize,
    }),
    //
    //обработка ошибок
    onError: ({ error }) => {
      // Log the error
      console.error(error);
    },
    onCatch: ({ error, errorInfo }) => {
      // Log the error
      console.error(error);
    },
    errorComponent: ({ error, reset }) => {
      const router = useRouter();

      return (
        <div>
          {error.message}
          <button
            onClick={() => {
              // ревалидировать после ошибки
              router.invalidate();
            }}
          >
            retry
          </button>
        </div>
      );
    },
    // актуальность данных
    staleTime: 0, //Infinity - отключение кеширования
    defaultStaleTime: 0,
    // актуальность предзагруженных данных
    preloadStaleTime: 30,
    defaultPreloadStaleTime: 30,
    //хранение перед удалением gc
    gcTime: 30 * 60 * 60,
    defaultGcTime: 30 * 60 * 60,
    //отказ от кеширования
    shouldReload: false,
    //оптимистическое пороговое значение
    pendingMs: 1,
    defaultPendingMs,
    pendingMinMs: 500,
    defaultPendingMinMs,
    //управление скроллом
    scrollToTopSelectors: ["#main-scrollable-area"], //id к которому нужно прокрутить
    scrollRestoration: true, //восстановление позиции прокрутки
    getScrollRestorationKey: (location) => location.pathname,
    scrollRestorationBehavior: "instant",
  }
);
```

# staticData

Можно передать в createFileRoute статические данные

```tsx
//как положить
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/posts")({
  staticData: {
    customData: "Hello!",
  },
});
```

```tsx
//как получить
import { createRootRoute } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: () => {
    const matches = useMatches();

    return (
      <div>
        {matches.map((match) => {
          return <div key={match.id}>{match.staticData.customData}</div>;
        })}
      </div>
    );
  },
});
```

Типизация

```ts
declare module "@tanstack/react-router" {
  interface StaticDataRouteOption {
    customData?: string; //сделать необязательным
  }
}
```
