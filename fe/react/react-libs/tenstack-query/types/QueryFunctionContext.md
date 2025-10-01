# QueryFunctionContext

Объект с полями:

- queryKey: QueryKey
- client: [QueryClient](../function/queryClient.md)
- signal?: AbortSignal
- meta: Record<string, unknown> | undefined

для Infinite Queries:

- pageParam: TPageParam
- direction: 'forward' | 'backward'
