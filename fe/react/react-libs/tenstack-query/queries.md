# Параллельные запросы

Можно через отдельные запросы

```tsx
function App () {
  // The following queries will execute in parallel
  const usersQuery = useQuery({ queryKey: ['users'], queryFn: fetchUsers })
  const teamsQuery = useQuery({ queryKey: ['teams'], queryFn: fetchTeams })
  const projectsQuery = useQuery({ queryKey: ['projects'], queryFn: fetchProjects })
  ...
}
```

[Можно через useQueries](./hooks/useQueries.md)

В случае цепочки зависимых запросов

```ts
const usersMessages = useQueries({
  queries: userIds
    ? userIds.map((id) => {
        return {
          queryKey: ["messages", id],
          queryFn: () => getMessagesByUsers(id),
        };
      })
    : [], // if userIds is undefined, an empty array will be returned
});
```

# работа с потерей фокуса

```tsx
focusManager.setEventListener((handleFocus) => {
  // Listen to visibilitychange
  if (typeof window !== "undefined" && window.addEventListener) {
    const visibilitychangeHandler = () => {
      handleFocus(document.visibilityState === "visible");
    };
    window.addEventListener("visibilitychange", visibilitychangeHandler, false);
    return () => {
      // Be sure to unsubscribe if a new handler is set
      window.removeEventListener("visibilitychange", visibilitychangeHandler);
    };
  }
});
```

# Повторные запросы

По умолчанию - 3

Настройка задержки

```tsx
import {
  QueryCache,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
});

function App() {
  return <QueryClientProvider client={queryClient}>...</QueryClientProvider>;
}
```

# Бесконечные запросы

если нужно проигнорировать в [useInfiniteQuery](./hooks/useInfiniteQuery.md) первую страницу

```ts
queryClient.setQueryData(["projects"], (data) => ({
  pages: data.pages.slice(1),
  pageParams: data.pageParams.slice(1),
}));
```

или любую другую

```ts
const newPagesArray =
  oldPagesArray?.pages.map((page) =>
    page.filter((val) => val.id !== updatedId)
  ) ?? [];

queryClient.setQueryData(["projects"], (data) => ({
  pages: newPagesArray,
  pageParams: data.pageParams,
}));
```

Если нет параметра пагинации в api

```ts
return useInfiniteQuery({
  queryKey: ["projects"],
  queryFn: fetchProjects,
  initialPageParam: 0,
  getNextPageParam: (lastPage, allPages, lastPageParam) => {
    if (lastPage.length === 0) {
      return undefined;
    }
    return lastPageParam + 1;
  },
  getPreviousPageParam: (firstPage, allPages, firstPageParam) => {
    if (firstPageParam <= 1) {
      return undefined;
    }
    return firstPageParam - 1;
  },
});
```
