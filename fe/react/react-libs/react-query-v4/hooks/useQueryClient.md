[queryCache](../instance/QueryCache.md)
[mutationCache](../instance/MutationCache.md)

```ts
const queryClient: QueryClient = useQueryClient({
  queryCache,
  mutationCache,
  logger,
  defaultOptions: {
    queries: {
      staleTime: Infinity,
    },
  },
});

type QueryClient = {
  fetchQuery: (any) => Promise<any>; //позволяет асинхронно сделать запрос и закешировать данные
  fetchInfiniteQuery: (any) => Promise<any>;
  prefetchQuery;
  prefetchInfiniteQuery;
  getQueryData; // queryClient.getQueryData(queryKey) дял получения кеша
  ensureQueryData; // const data = await queryClient.ensureQueryData({ queryKey, queryFn }) если существуют данные
  getQueriesData;
  setQueryData; // queryClient.setQueryData(queryKey, updater) синхронная операция fetchQuery - нет
  getQueryState;
  setQueriesData;
  invalidateQueries;
  refetchQueries;
  cancelQueries;
  removeQueries;
  resetQueries;
  isFetching;
  isMutating;
  getLogger;
  getDefaultOptions;
  setDefaultOptions;
  getQueryDefaults;
  setQueryDefaults;
  getMutationDefaults;
  setMutationDefaults;
  getQueryCache;
  getMutationCache;
  clear;
  resumePausedMutations;
};
```
