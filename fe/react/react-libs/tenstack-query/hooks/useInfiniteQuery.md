# useInfiniteQuery

- если данные устареют то перезагрузку данных будет идти параллельно

```tsx
const {
  data: {
    pages: [],
    pageParams: {},
  },
  fetchNextPage,
  fetchPreviousPage,
  hasNextPage,
  hasPreviousPage,
  isFetchingNextPage,
  isFetchingPreviousPage,
} = useInfiniteQuery({
  initialPageParam: 0,
  getNextPageParam: (lastPage, pages) => lastPage.nextCursor,
  getPreviousPageParam: (firstPage, pages) => firstPage.prevCursor,
  // лимит по страницам
  maxPages: 3,
  // в случае если данные нужны в обратном порядке
  select: (data) => ({
    pages: [...data.pages].reverse(),
    pageParams: [...data.pageParams].reverse(),
  }),
});
```

# Параметры:

Все те же самые параметры что и [useQuery](useQuery.md) c полями:

## queryFn

```tsx
type QueryFn = (context: QueryFunctionContext) => Promise<TData>;
```

[QueryFunctionContext c учетом что это InfiniteQuery](../types/QueryFunctionContext.md)

## initialPageParam

обязательный - TPageParam

## getNextPageParam

обязательный

```ts
type GetNextPageParam = (
  lastPage,
  allPages,
  lastPageParam,
  allPageParams
) => TPageParam | undefined | null;
```

## getNextPageParam

```ts
type GetPreviousPageParam = (
  firstPage,
  allPages,
  firstPageParam,
  allPageParams
) => TPageParam | undefined | null;
```

## maxPages

number | undefined - максимальное количество страниц хранящихся в кеше

# Возвращает

data.pages: TData[]

- data.pageParams: unknown[]
- isFetchingNextPage: boolean
- isFetchingPreviousPage: boolean
- fetchNextPage

```ts
type fetchNextPage: (options?: FetchNextPageOptions) => Promise<UseInfiniteQueryResult>
```

- fetchPreviousPage

```ts
type FetchPreviousPage: (options?: FetchPreviousPageOptions) => Promise<UseInfiniteQueryResult>
```

- hasNextPage: boolean
- hasPreviousPage: boolean
- isFetchNextPageError: boolean
- isFetchPreviousPageError: boolean
- isRefetching: boolean
- isRefetchError: boolean
- promise:

```ts
TPromise = Promise<TData>;
```
