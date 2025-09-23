# RouteOptions

# свойства

## path

строка для определения пути

## id

строка, обязательно если нет path

## component

RouteComponent или LazyRouteComponent (опциональный, Outlet по умолчанию ) - компонент, который будет отражаться

## errorComponent

RouteComponent or LazyRouteComponent (опциональный, routerOptions.defaultErrorComponent по умолчанию )

## pendingComponent

RouteComponent or LazyRouteComponent (опциональный, routerOptions.defaultPendingComponent по умолчанию ) - компонент, который буде отображаться при ожидании или достижения pendingMs

## notFoundComponent

RouteComponent or LazyRouteComponent (опциональный, routerOptions.defaultNotFoundComponent по умолчанию ) - компонент, при notfound

## search

### search.middlewares

(({search: TSearchSchema, next: (newSearch: TSearchSchema) => TSearchSchema}) => TSearchSchema)[]

промежуточный обработчики для строки поиска

```tsx
import { z } from "zod";
import { createFileRoute } from "@tanstack/router";

export const Route = createFileRoute("/users")({
  component: UsersPage,
  validateSearch: z.object({
    page: z.number().min(1).default(1),
    q: z.string().optional(),
  }),
  search: {
    middlewares: [
      // 1) нормализация пустых строк
      ({ search }) => {
        if (search.q === "") {
          return { ...search, q: undefined };
        }
        return search;
      },
      // 2) ограничение номера страницы
      ({ search }) => {
        if (search.page && search.page > 100) {
          return { ...search, page: 100 };
        }
        return search;
      },
    ],
  },
});
```

### search.validate

### search.parse

### search.stringify

## staleTime

число - время на сколько актуальные данные маршрута

## preloadStaleTime

число 30_000 - данные актуальные при пред-загрузке

## gcTime

число - 30 минут - хранение данных по маршруту

## shouldReload

boolean | ((args: LoaderArgs) => boolean)

будут ли перезагружаться данные

## caseSensitive

boolean - маршрут будет сопоставляться с учетом регистра.

## wrapInSuspense

boolean - принудительная обертка в suspense

## pendingMs

number - defaultPendingMs - сколько будет отображаться pendingComponent

## pendingMinMs

number - предотвратить мигание ожидающего компонента на экране на долю секунды

## preloadMaxAge

number - сколько будут кэшироваться пред-загруженные данные маршрута

## codeSplitGroupings

```ts
type CodeSplitGroupings = Array<
  Array<
    | "loader"
    | "component"
    | "pendingComponent"
    | "notFoundComponent"
    | "errorComponent"
  >
>;
```

разбиение по чанкам компонентов

<!-- методы -------------------------------------------------------------->

# методы

## getParentRoute

() => [TParentRoute](./TParentRoute.md) - возвращает родительски роут компонент

## validateSearch

(rawSearchParams: unknown) => TSearchSchema или (searchParams: TSearchSchemaInput & SearchSchemaInput) => TSearchSchema - опционально, если функция пробросит ошибку, то отобразиться errorComponent

## parseParams

(params: TParams) => Record<string, string> (обязательный если есть parseParams) - для парсинга строк поиска

## params

### params.parse

Type: (rawParams: Record<string, string>) => TParams для получения строки

### params.stringify

Type: (params: TParams) => Record<string, string> дял получения параметров в виде объекта

## beforeLoad

```ts
type beforeLoad = (
  opts: RouteMatch & {
    search: TFullSearchSchema;
    abortController: AbortController;
    preload: boolean;
    params: TAllParams;
    context: TParentContext;
    location: ParsedLocation;
    buildLocation: BuildLocationFn<AnyRoute>;
    cause: "enter" | "stay";
  }
) => Promise<TRouteContext> | TRouteContext | void;
```

опционально, если вернет промис, то состояние будет pending

## loader

```ts
type loader = (
  opts: RouteMatch & {
    abortController: AbortController;
    cause: "preload" | "enter" | "stay";
    context: TAllContext;
    deps: TLoaderDeps;
    location: ParsedLocation;
    params: TAllParams;
    preload: boolean;
    parentMatchPromise: Promise<MakeRouteMatchFromRoute<TParentRoute>>;
    navigate: NavigateFn<AnyRoute>; // @deprecated
    route: AnyRoute;
  }
) => Promise<TLoaderData> | TLoaderData | void;
```

[RouteMatch](./RouteMatch.md)

- ориентируется на состояние промиса
- - при ожидании pendingComponent
- - при ошибки onError
- если возвращает TLoaderData данные будут доступны в useLoaderData

## loaderDeps

```ts
type loaderDeps = (opts: { search: TFullSearchSchema }) => Record<string, any>;
```

Функция которая должна возвращать сериализуемые данные, будут доступны в deps. Нужен, если у тебя есть внешние зависимости, не привязанные напрямую к маршруту.

```tsx
//- Пользователь логинится → window.auth.userId меняется.
//- loaderDeps вернёт новый userId.
//- Router заметит изменение и снова вызовет loader.
export const Route = createFileRoute("/profile")({
  component: ProfilePage,
  loader: async ({ deps }) => {
    // deps.userId попадёт сюда
    return getUser(deps.userId);
  },
  loaderDeps: () => {
    // например, берем userId из глобального стора auth
    return { userId: window.auth.userId };
  },
});
```

## remountDeps

```ts
type remountDeps = (opts: RemountDepsOptions) => any;

interface RemountDepsOptions<
  in out TRouteId,
  in out TFullSearchSchema,
  in out TAllParams,
  in out TLoaderDeps
> {
  routeId: TRouteId;
  search: TFullSearchSchema;
  params: TAllParams;
  loaderDeps: TLoaderDeps;
}
```

функция для определения пере-рендера роута

remountDeps: ({ params }) => params

## onError

(error: any) => void - вызывается при ошибке

## onEnter

(совпадение: RouteMatch) => void - вызывается когда маршрут найден

## onStay

(совпадение: RouteMatch) => void - при сопоставлении маршрута

## onLeave

(совпадение: RouteMatch) => void - при несовпадении

## onCatch

(ошибка: Ошибка, errorInfo: ErrorInfo) => void - при обнаружении ошибок

## headers

```ts
type headers = (opts: {
  matches: Array<RouteMatch>;
  match: RouteMatch;
  params: TAllParams;
  loaderData?: TLoaderData;
}) => Promise<Record<string, string>> | Record<string, string>;
```

нужно для SSR

## method

```ts
type head = (ctx: {
  matches: Array<RouteMatch>;
  match: RouteMatch;
  params: TAllParams;
  loaderData?: TLoaderData;
}) =>
  | Promise<{
      links?: RouteMatch["links"];
      scripts?: RouteMatch["headScripts"];
      meta?: RouteMatch["meta"];
      styles?: RouteMatch["styles"];
    }>
  | {
      links?: RouteMatch["links"];
      scripts?: RouteMatch["headScripts"];
      meta?: RouteMatch["meta"];
      styles?: RouteMatch["styles"];
    };
```

## scripts

```ts
type scripts = (ctx: {
  matches: Array<RouteMatch>;
  match: RouteMatch;
  params: TAllParams;
  loaderData?: TLoaderData;
}) => Promise<RouteMatch["scripts"]> | RouteMatch["scripts"];
```

как headers только для скриптов
