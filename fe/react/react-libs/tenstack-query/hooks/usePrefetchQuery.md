# usePrefetchQuery

Позволяет пред-загрузить запрос

```tsx
import { usePrefetchQuery } from "@tanstack/react-query";

function UserListItem({ userId }) {
  const prefetchUser = usePrefetchQuery({
    queryKey: ["user", userId],
    queryFn: () => fetchUser(userId),
    staleTime: 1000 * 60,
    enabled: false, // не выполняется сразу
  });

  return (
    <div onMouseEnter={() => prefetchUser()} onFocus={() => prefetchUser()}>
      Наведи, чтобы заранее загрузить профиль
    </div>
  );
}
```

```tsx
function Posts({ page }) {
  const query = useQuery({
    queryKey: ["posts", page],
    queryFn: () => fetchPosts(page),
  });

  const prefetchNextPage = usePrefetchQuery({
    queryKey: ["posts", page + 1],
    queryFn: () => fetchPosts(page + 1),
  });

  useEffect(() => {
    if (query.data?.hasNextPage) {
      prefetchNextPage();
    }
  }, [query.data]);

  return <PostList posts={query.data?.items} />;
}
```

# usePrefetchInfiniteQuery
