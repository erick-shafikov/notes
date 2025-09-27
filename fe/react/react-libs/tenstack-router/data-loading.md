# загрузка данных

Существуют следующие варианты загрузки данных:

- [опция loader в createFileRoute](./functions/createFileRoute.md#функция-loader)
- [функция getRouteApi](./functions/getRouteApi.md)

# жизненный цикл

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

в рамках TSQ существует встроенное кеширование

# api в контексте

возможно передать функцию по загрузке данных через контекст

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
```

# принудительная пред-загрузка

Принудительную пред-загрузку можно реализовать с помощью хука useRouter:

- [preloadRoute](./hooks/useRouter.md)

# статические данные

[Можно передать в createFileRoute статические данные](./functions/createFileRoute.md#staticdata)

# разделение loader-а

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

# взаимодействие с TSQ

```tsx
// src/routes/posts.$postId.tsx
import { createFileRoute } from "@tanstack/react-router";
import { useSuspenseQuery } from "@tanstack/react-query";
import { slowDataOptions, fastDataOptions } from "~/api/query-options";

export const Route = createFileRoute("/posts/$postId")({
  loader: async ({ context: { queryClient } }) => {
    // начните загрузку, но не дожидайтесь ее если это долгие данные
    queryClient.prefetchQuery(slowDataOptions());

    // дожидайтесь данные которые быстро загружаются
    await queryClient.ensureQueryData(fastDataOptions());
  },
  component: PostIdComponent,
});

function PostIdComponent() {
  const fastData = useSuspenseQuery(fastDataOptions());

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
