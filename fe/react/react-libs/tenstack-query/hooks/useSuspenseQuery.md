# useSuspenseQuery

Параметры:

- Аргументы:
- - все что и useQuery но без throwOnError, enabled, placeholderData
- возвращает:
- все что и useQuery но data всегда есть, нет isPlaceholderData , status всегда success или error

```tsx
function UserProfile({ id }) {
  const user = useSuspenseQuery({
    queryKey: ["user", id],
    queryFn: () => fetchUser(id),
  });

  return <div>{user.name}</div>;
}

function App() {
  return (
    <Suspense fallback={<p>Загрузка...</p>}>
      <ErrorBoundary fallback={<p>Ошибка!</p>}>
        <UserProfile id={42} />
      </ErrorBoundary>
    </Suspense>
  );
}
```

# useSuspenseInfiniteQuery

аналогично

# useSuspenseQueries
