# useCanGoBack

```tsx
import { useRouter, useCanGoBack } from "@tanstack/react-router";

function Component() {
  const router = useRouter();
  const canGoBack = useCanGoBack();

  return (
    <div>
      {canGoBack ? (
        <button onClick={() => router.history.back()}>Go back</button>
      ) : null}
    </div>
  );
}
```
