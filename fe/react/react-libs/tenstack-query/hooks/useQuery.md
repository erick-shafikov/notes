# useQuery

# Параметры, принимает:

- принимает 2 параметра

## объект опций

### queryKey

```ts
type QueryKey = unknown[];
```

Ключи объекта

### queryFn

```ts
type QueryFn = (context: QueryFunctionContext) => Promise<TData>;
```

- должна возвращать промис

[контекст который принимает QueryFunctionContext](../types/QueryFunctionContext.md)

использование signal

```js
const query = useQuery({
  queryKey: ["todos"],
  queryFn: async ({ signal }) => {
    const todosResponse = await fetch("/todos", {
      // Pass the signal to one fetch
      signal,
    });
    const todos = await todosResponse.json();

    const todoDetails = todos.map(async ({ details }) => {
      const response = await fetch(details, {
        // Or pass it to several
        signal,
      });
      return response.json();
    });

    return Promise.all(todoDetails);
  },
});
```

### gcTime

30_000 - сколько по времени данные хранятся в кеше

### enabled

true - активный ли запрос

### networkMode

```ts
type NetworkMode = "online" | "always" | "offlineFirst";
```

### initialData

Хранится в кеше

```ts
type InitialData = TData | () => TData
```

значение по умолчанию для кеша данного запроса

### initialDataUpdatedAt

```ts
type InitialDataUpdatedAt = number | (() => number | undefined);
```

### meta

```ts
type Meta = Record<string, unknown>;
```

данные для QueryFunctionContext поля meta

### notifyOnChangeProps

```ts
type NotifyOnChangeProps =
  | string[]
  | "all"
  | (() => string[] | "all" | undefined);
```

если указаны параметры то компонент будет перерисован, если изменился один из параметров

### placeholderData

```ts
type PlaceholderData = TData | (previousValue: TData | undefined, previousQuery: Query | undefined) => TData
```

данные которые будут отображаться при pending статусе, не хранятся в кеше

```jsx
function Todos() {
  // стоит использовать useMemo
  const placeholderData = useMemo(() => generateFakeTodos(), []);
  const result = useQuery({
    queryKey: ["todos"],
    queryFn: () => fetch("/todos"),
    placeholderData,
  });
}
```

### queryKeyHashFn

```ts
type QueryKeyHashFn = (queryKey: QueryKey) => string;
```

для хеширования ключей

### refetchInterval

```ts
type RefetchInterval =
  | number
  | false
  | ((query: Query) => number | false | undefined);
```

### refetchIntervalInBackground

```ts
type RefetchIntervalInBackground =
  | boolean
  | "always"
  | ((query: Query) => boolean | "always");
```

будут ли происходить пере-запрос на фоне

### refetchOnMount

### refetchOnReconnect

```ts
type RefetchOnReconnect =
  | boolean
  | "always"
  | ((query: Query) => boolean | "always");
```

### refetchOnWindowFocus

```ts
type RefetchOnWindowFocus =
  | boolean
  | "always"
  | ((query: Query) => boolean | "always");
```

### retry

```ts
type Retry = boolean | number | (failureCount: number, error: TError) => boolean
```

### retryOnMount

true - если false, то не будет активность при монтировании в случае ошибки

### retryDelay

```ts
type retryDelay = number | (retryAttempt: number, error: TError) => number
```

### select

```ts
type Select = (data: TData) => unknown;
```

### staleTime

```ts
type staleTime = number | "static" | ((query: Query) => number | "static");
```

0 - через сколько данные будут являться устаревшими

### structuralSharing

```ts
type StructuralSharing = boolean | (oldData: unknown | undefined, newData: unknown) => unknown
```

true - если false, то сопоставление объектов при новых данных получаемых по запросу будет отключено, что приведет к ре-рендерам

### subscribed

true - если false то useQuery отпишется от кеша

### throwOnError

```ts
type throwOnError = undefined | boolean | (error: TError, query: Query) => boolean
```

- true - если нужно прокинуть ошибку до ближайшего error boundary
- false - сброс

## Второй параметр

### queryClient

если не указан, то возьмет ближайший контекст

<!--  -->

# Возвращает:

Объект с полями:

## data

```ts
type Data = TData | undefined;
```

## dataUpdatedAt

number - когда последний раз был статус "success"

## error

```ts
type Error = null | TError;
```

## errorUpdatedAt

number - когда последний раз был статус "error"

## failureCount

number - количество ошибок

## failureReason

```ts
type FailureReason = null | TError;
```

## fetchStatus

```ts
type FetchStatus = "fetching" | "paused" | "idle";
```

## isError

boolean

## isFetched

boolean - если был запрос

## isFetchedAfterMount

boolean - может использоваться для отображения предыдущих данных

## isFetching

boolean

## isInitialLoading

boolean

## isLoading

boolean === isFetching && isPending

## isLoadingError

boolean

## isPaused

boolean

## isPending

boolean

## isPlaceholderData

boolean - если отображается PlaceholderData

## isRefetchError

boolean - если ошибка при Refetch

## isRefetching

boolean === isFetching && !isPending

## isStale

boolean

## isSuccess

boolean

## isEnabled

boolean

## promise

```ts
type Promise = Promise<TData>;
```

промис который разрешиться при получении данных, нужен experimental_prefetchInRender флаг в QueryClient

```jsx
import React from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchTodos, type Todo } from "./api";

function TodoList({ query }: { query: UseQueryResult<Todo[]> }) {
  const data = React.use(query.promise);

  return (
    <ul>
      {data.map((todo) => (
        <li key={todo.id}>{todo.title}</li>
      ))}
    </ul>
  );
}

export function App() {
  const query = useQuery({ queryKey: ["todos"], queryFn: fetchTodos });

  return (
    <>
      <h1>Todos</h1>
      <React.Suspense fallback={<div>Loading...</div>}>
        <TodoList query={query} />
      </React.Suspense>
    </>
  );
}
```

## refetch

```ts
type Refetch = (options: {
  throwOnError: boolean;
  cancelRefetch: boolean;
}) => Promise<UseQueryResult>;
```

## status

```ts
type Status = "pending" | "error" | "success";
```
