Встроенное кеширование:

- встроенное кеширование

Жизненный цикл роута:

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

# взаимодействие с tenstack-query

```tsx
// src/routes/posts.$postId.tsx
import { createFileRoute } from "@tanstack/react-router";
import { useSuspenseQuery } from "@tanstack/react-query";
import { slowDataOptions, fastDataOptions } from "~/api/query-options";

export const Route = createFileRoute("/posts/$postId")({
  loader: async ({ context: { queryClient } }) => {
    // Kick off the fetching of some slower data, but do not await it
    queryClient.prefetchQuery(slowDataOptions());

    // Fetch and await some data that resolves quickly
    await queryClient.ensureQueryData(fastDataOptions());
  },
  component: PostIdComponent,
});

function PostIdComponent() {
  const fastData = useSuspenseQuery(fastDataOptions());

  // do something with fastData

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <SlowDataComponent />
    </Suspense>
  );
}

function SlowDataComponent() {
  const data = useSuspenseQuery(slowDataOptions());

  return <div>{data}</div>;
}
```

```tsx
export const Route = createFileRoute("/posts/$postId/deep")({
  // плохо
  loader: ({ context: { queryClient }, params: { postId } }) =>
    queryClient.ensureQueryData(postQueryOptions(postId)),
  component: PostDeepComponent,
  // хорошо, выведет точный тип без промиса
  loader: async ({ context: { queryClient }, params: { postId } }) => {
    await queryClient.ensureQueryData(postQueryOptions(postId));
  },
});
function PostDeepComponent() {
  const params = Route.useParams();
  const data = useSuspenseQuery(postQueryOptions(params.postId));

  return <></>;
}
```

принудительная предзагрузка

```tsx
function Component() {
  const router = useRouter();

  useEffect(() => {
    async function preload() {
      try {
        const matches = await router.preloadRoute({
          to: postRoute,
          params: { id: 1 },
        });
      } catch (err) {
        // Failed to preload route
      }
    }

    preload();
  }, [router]);

  // несколько

  useEffect(() => {
    async function preloadRouteChunks() {
      try {
        const postsRoute = router.routesByPath["/posts"];
        await Promise.all([
          router.loadRouteChunk(router.routesByPath["/"]),
          router.loadRouteChunk(postsRoute),
          router.loadRouteChunk(postsRoute.parentRoute),
        ]);
      } catch (err) {
        // Failed to preload route chunk
      }
    }

    preloadRouteChunks();
  }, [router]);

  return <div />;
}
```
