# useQueries

```ts
const someArrayDataToFetch = [];

const {} = useQueries({
  queries: someArrayDataToFetch.map(({ id }) => ({
    queryKey: ["some", id],
    queryFn: () => fetchFunction(id),
  })),
});
```

Параметры:

- Принимает (объект):
- - queries - объект из [UseQueryOptions](useQuery.md#параметры-принимает) только без параметра placeholderData

- - queryClient - [QueryClient](../function/queryClient.md)
- - combine - для комбинирования результатов

Запустится если один из результатов запроса меняется

```ts
type Combine = (result: UseQueriesResults) => TCombinedResult;
```

```ts
const ids = [1, 2, 3];
const combinedQueries = useQueries({
  queries: ids.map((id) => ({
    queryKey: ["post", id],
    queryFn: () => fetchPost(id),
  })),
  combine: (results) => {
    return {
      data: results.map((result) => result.data),
      pending: results.some((result) => result.isPending),
    };
  },
});
```

- Возвращает [UseQueryResult](useQuery.md#возвращает)
