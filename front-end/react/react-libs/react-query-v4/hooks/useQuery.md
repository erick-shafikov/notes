```tsx
const {
  data, // TData | undefined данные, по умолчанию undefined
  dataUpdatedAt, //timestamp последний раз когда был status === success
  error, //null по умолчанию
  errorUpdateCount, //
  errorUpdatedAt, //timestamp последний раз когда был status === error
  failureCount, //
  failureReason, //
  fetchStatus, // fetching | paused | idle
  isError, //boolean если status === error
  isFetched, //boolean если запрос произведен
  isFetchedAfterMount, //boolean
  isFetching, //boolean
  isInitialLoading, //boolean  === isFetching && isLoading
  isLoading, //boolean если status === loading
  isLoadingError, //boolean если ошибка при первом запросе
  isPaused, //boolean
  isPlaceholderData, //boolean если показывается PlaceholderData
  isPreviousData, //boolean
  isRefetchError, //boolean если ошибка при повторном запросе
  isRefetching, //boolean
  isStale, //boolean
  isSuccess, //boolean если status === success
  refetch, //  (options: { throwOnError: boolean, cancelRefetch: boolean }) => Promise<UseQueryResult>
  remove, // () => void удалить из кеша функцию
  status, //loading | error | success - статус запроса
} = useQuery({
  queryKey, // unknown[] массив ключей
  queryFn, //queryFn: (context: QueryFunctionContext) => Promise<TData> функция запроса, которая должна возвращать промис
  // --------------------------------------------------------------------
  cacheTime, // number | Infinity время хранения данных в кеше
  enabled, // enabled: boolean активна ли функция запроса
  networkMode, //'online' | 'always' | 'offlineFirst режим работы при состоянии сети
  initialData, // initialData: TData | () => TData закешированные начальные данные
  initialDataUpdatedAt, // number | (() => number | undefined) in milliseconds - обновление начальных данных
  keepPreviousData, // boolean
  meta, //Record<string, unknown> - QueryFunctionContext.meta - будет доступно как аргумент функции в queryFn
  notifyOnChangeProps, //string[] | "all" | (() => string[] | "all")
  onError, //deprecated
  onSettled, //deprecated
  onSuccess, //deprecated
  placeholderData, // TData | () => TData данные по умолчанию, не кешируются
  queryKeyHashFn, //(queryKey: QueryKey) => string позволяет захешировать ключ в строку
  refetchInterval, // number | false | ((data: TData | undefined, query: Query) => number | false) - мс, частота с которой будут повторяться запросы
  refetchIntervalInBackground, //refetchIntervalInBackground: boolean
  refetchOnMount, // boolean | "always" | ((query: Query) => boolean | "always")
  refetchOnReconnect, //boolean | "always" | ((query: Query) => boolean | "always")
  refetchOnWindowFocus, //boolean | "always" | ((query: Query) => boolean | "always")
  retry, //boolean | number | (failureCount: number, error: TError) => boolean количество повторных запросов при ошибке
  retryOnMount, //boolean
  retryDelay, //retryDelay: number | (retryAttempt: number, error: TError) => number количество мс для повторных запросов в случае ошибки
  select, //select: (data: TData) => unknown преобразовать данные
  staleTime, //number | Infinity - время, которое определяет устаревшие данные или нет
  structuralSharing, //boolean | ((oldData: TData | undefined, newData: TData) => TData) - комбинировать старые и новые данные
  suspense, //boolean
  useErrorBoundary, //undefined | boolean | (error: TError, query: Query) => boolean
  context, //React.Context<QueryClient | undefined> при использовании кастомного контекста
});
```

# QueryFunctionContext

```tsx
type QueryFunctionContext = {
  queryKey: QueryKey; //ключи unknown[]
  pageParam?: unknown;
  // only for Infinite Queries
  // the page parameter used to fetch the current page
  signal?: AbortSignal;
  // AbortSignal instance provided by TanStack Query
  // Can be used for Query Cancellation
  meta: Record<string, unknown> | undefined;
  // an optional field you can fill with additional information about your query
};
```
