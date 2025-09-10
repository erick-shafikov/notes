# createMemoryHistory

Позволяет передать историю напрямую в Route, используется там, где нет истории браузера (SSR)

```tsx
import { createMemoryHistory, createRouter } from "@tanstack/react-router";

const memoryHistory = createMemoryHistory({
  initialEntries: ["/"], // Pass your initial url
});

const router = createRouter({ routeTree, history: memoryHistory });
```
