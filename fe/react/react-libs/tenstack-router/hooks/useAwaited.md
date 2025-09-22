# useAwaited

ждет пока промис не прокинет исключение, либо не разрешиться

```tsx
import { useAwaited } from "@tanstack/react-router";

function Component() {
  const { deferredPromise } = route.useLoaderData();

  const data = useAwaited({
    // принимает 1 параметр
    promise: myDeferredPromise,
  });
  // ...
}
```
