# skipToken

говорит о том, что нужно не выполнять запрос

```tsx
import { useQuery, skipToken } from "@tanstack/react-query";

function User({ userId }: { userId?: string }) {
  const query = useQuery({
    queryKey: ["user", userId],
    queryFn: userId
      ? () => fetch(`/api/users/${userId}`).then((res) => res.json())
      : skipToken, // 👈 запрос не выполнится
  });

  return <div>{JSON.stringify(query.data)}</div>;
}
```
