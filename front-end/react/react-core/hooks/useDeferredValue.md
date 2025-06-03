## useDeferredValue

useDeferredValue(value) – хук, который позволяет отложить изменение, это значение должно быть примитивом
интеграция со Suspense, пользователь не увидит fallback а будет видеть предыдущее значение

```jsx
import { useState, useDeferredValue } from "react";

function SearchPage() {
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  // ...
}
```
