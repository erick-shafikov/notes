# dehydrate

Создает кеш для пред-запроса с сервера

Параметры:

- принимает аргументы:
- - client
- - options (объект):
- - - shouldDehydrateMutation: (mutation: Mutation) => boolean
- - - shouldDehydrateQuery: (query: Query) => boolean
- - - serializeData?: (data: any) => any
- - - shouldRedactErrors?: (error: unknown) => boolean
- возвращает dehydratedState

```ts
import { dehydrate } from "@tanstack/react-query";

const dehydratedState = dehydrate(queryClient, {
  shouldDehydrateQuery,
  shouldDehydrateMutation,
});
```

# hydrate

Параметры:

- принимает:
- - client: QueryClient
- - dehydratedState: DehydratedState
- - options: HydrateOptions
- - - defaultOptions: DefaultOptions
- - - queryClient?: QueryClient
