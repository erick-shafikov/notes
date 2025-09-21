# isNotFound

Проверка obj на предмет NotFound ошибки

```tsx
import { isNotFound } from "@tanstack/react-router";

function somewhere(obj: unknown) {
  if (isNotFound(obj)) {
    // ...
  }
}
```
