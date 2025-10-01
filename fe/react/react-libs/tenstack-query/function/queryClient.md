# queryClient

Позволяет взаимодействовать с query клиентом

```ts
import { QueryClient } from "@tanstack/react-query";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: Infinity,
    },
  },
});

await queryClient.prefetchQuery({ queryKey: ["posts"], queryFn: fetchPosts });
```

# класс принимает

- Принимает (объект):
- - queryCache? - объект с кешем вида [QueryCache](./queryCache.md)
- - mutationCache? - объект кеша мутаций [MutationCache](./mutationCache.md)
- - defaultOptions? - опции по умолчанию для всех query и мутаций, используется для гидратации

# методы экземпляра

## fetchQuery

асинхронный, вернет результат. Параметры:

- принимает на вход все тоже что и [useQuery](../hooks/useQuery.md)
- вернет Promise от TData

## fetchInfiniteQuery

асинхронный, тоже самое что и fetchQuery

Параметры такие же как и у fetchQuery

```ts
await queryClient.prefetchQuery({ queryKey });
```

## prefetchQuery

асинхронный, тоже самое что и fetchQuery, но оне возвращает результат

```ts
try {
  const data = await queryClient.fetchQuery({
    queryKey,
    queryFn,
    staleTime: 10000,
  });
} catch (error) {
  console.log(error);
}
```

## prefetchInfiniteQuery

асинхронный, тоже самое что и fetchInfiniteQuery, но оне возвращает результат

## getQueryData

синхронный, вернет данные если есть в кеше, в противном случае undefined, параметры:

- принимает queryKey
- возвращает TData | undefined

## ensureQueryData

асинхронный, вернет данные если есть, если нет вызывается fetchQuery и возвращается результата, параметры:

- принимает:
- - опции fetchQuery
- - revalidateIfStale? - false , при true вернет закешированные данные сразу
- возвращает Promise от TData

## ensureInfiniteQueryData

асинхронный ensureQueryData для infinite query

## getQueriesData

синхронный, вернет данные для нескольких запросов, параметры:

- принимает queryKeys
- возвращает массив вида [queryKey: QueryKey, data: TQueryFnData | undefined][]

## setQueryData

синхронный, для обновления данных в кеше

- сбросятся через 5 минут
- должен быть иммутабельным

параметры:

- принимает (объект):
- - queryKey: QueryKey
- - updater: TQueryFnData | undefined | ((oldData: TQueryFnData | undefined) => TQueryFnData | undefined) - если функция есть возможность получить предыдущие данные, если объект, то заменит

## getQueryState

асинхронный, вернет состояние запроса

## setQueriesData

синхронный, для обновления нескольких запросов, параметры:

- принимает (объект):
- - filters - ключи
- - updater: TQueryFnData | (oldData: TQueryFnData | undefined) => TQueryFnData

## invalidateQueries

обновит данные в кеше, позволяет обновить устаревшие и не обновлять свежие данные, параметры:

- принимает 2 аргумента:
- - filters(объект):
- - - queryKey - ключи
- - - updater: TQueryFnData | (oldData: TQueryFnData | undefined) => TQueryFnData
- - options(объект):
- - - throwOnError? - если true то пробросит ошибку если есть хоть одна
- - - cancelRefetch? - true, если false не будет запускать запрос, если он уже выполняется

## refetchQueries

для перезапуска запроса на определенных условиях, параметры:

- принимает:
- - filters
- - options
- - - throwOnError
- - - cancelRefetch
- вернет промис

## cancelQueries

позволяет отменить запрос, параметры:

- принимает:
- - фильтры
- - cancelOptions

## removeQueries

удаляет запросы из кеша, параметры:

- принимает:
- - фильтры

## resetQueries

сброс до изначальных, параметры:

- принимает:
- - filters
- - options
- - - throwOnError
- - - cancelRefetch
- возвращает:
- -промис

## getDefaultOptions

вернет опции по умолчанию

## setDefaultOptions

установит опции по умолчанию

## getQueryDefaults

вернет опции для определенного запроса

## setQueryDefaults

установит опции для запроса, параметры:

- принимает:
- - фильтры
- - QueryOptions

```ts
queryClient.setQueryDefaults(["posts"], { queryFn: fetchPosts });

function Component() {
  const { data } = useQuery({ queryKey: ["posts"] });
}
```

## getMutationDefaults

вернет опции для мутации

## setMutationDefaults

установит опции для мутации

## getQueryCache

вернет кеш

## getMutationCache

вернет кеш

## clear

удалит кеш

## resumePausedMutations

восстановит мутации, которые были отменены из-за проблем с сетью

# свойства экземпляра

## isFetching

вернет количество запрос на текущий момент, параметры:

- принимает фильтры
- возвращает число запросов

## isMutating

вернет количество запросов, параметры:

- принимает фильтры
- возвращает число запросов
