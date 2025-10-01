# streamedQuery

Позволяет работать с AsyncIterable данными

```ts
import { experimental_streamedQuery as streamedQuery } from "@tanstack/react-query";

const query = queryOptions({
  queryKey: ["data"],
  queryFn: streamedQuery({
    streamFn: fetchDataInChunks,
  }),
});
```

Параметры:

- Принимает (объект с полями):
- - streamFn: (context: QueryFunctionContext) => Promise от AsyncIterable TData
- - refetchMode? - 'append' | 'reset' | 'replace'
- - reducer?: (accumulator: TData, chunk: TQueryFnData) => TData
- - initialValue?: TData = TQueryFnData
