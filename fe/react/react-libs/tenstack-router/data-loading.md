Встроенное кеширование:

- встроенное кеширование

Жизненный цикл:

- Сопоставление маршрутов:
- - route.params.parse
- - route.validateSearch
- предварительная загрузка:
- - route.beforeLoad
- - route.onError
- - - route.errorComponent
- - - parentRoute.errorComponent
- - - router.defaultErrorComponent
- Загрузка маршрута (параллельная)
- - route.component.preload
- - route.loader:
- - - route.pendingComponent
- - - route.component
- - route.onError
- - - route.errorComponent
- - - parentRoute.errorComponent
- - - router.defaultErrorComponent

```tsx
//router.tsx
// передать контекст вниз по всем роутам [2]
const router = createRouter({
  routeTree,
  context: {
    // Supply the fetchPosts function to the router context
    fetchPosts,
  },
  errorComponent: ({ error }) => {
    if (error instanceof MyCustomError) {
      // отлов определенных ошибок
      return <div>{error.message}</div>;
    }

    // компонент ошибки по умолчанию
    return <ErrorComponent error={error} />;
  },
});
// page.tsx
import { getRouteApi } from "@tanstack/react-router"; // доступ к данным [1]

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
  }
);

//[1]
const routeApi = getRouteApi("/posts");

function PostComponent() {
  const {
    //получение данных
  } = Route.useLoaderData();

  //[1]
  const data = routeApi.useLoaderData();

  return <></>;
}
```
