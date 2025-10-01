# TimeoutManager

Управление таймерами

методы:

- setTimeoutProvider
  TimeoutProvider
- setTimeout
- clearTimeout
- setInterval
- clearInterval

```ts
import { timeoutManager, QueryClient } from "@tanstack/react-query";
import { CustomTimeoutProvider } from "./CustomTimeoutProvider";

timeoutManager.setTimeoutProvider(new CustomTimeoutProvider());

export const queryClient = new QueryClient();
```
