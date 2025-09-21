# isRedirect

Проверка obj на предмет redirect

```ts
import { isRedirect } from "@tanstack/react-router";

function somewhere(obj: unknown) {
  if (isRedirect(obj)) {
    // ...
  }
}
```
