## useDeferredValue

useDeferredValue(value) – хук, который позволяет отложить изменение, это значение должно быть примитивом
интеграция со Suspense, пользователь не увидит fallback а будет видеть предыдущее значение

```tsx
import { useState, useDeferredValue } from "react";

function SearchPage(): JSX.Element {
  const [query, setQuery] = useState<string>("");
  const deferredQuery = useDeferredValue(query);
  // ...
}
```
