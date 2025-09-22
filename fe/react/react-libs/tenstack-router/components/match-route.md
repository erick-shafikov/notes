# MatchRoute

Компонент, который повторяет логику [useMatchRoute](../hooks/useMatchRoute.md) позволяет отрисовать компонент, если совпадает путь

```tsx
import { MatchRoute } from "@tanstack/react-router";

function Component() {
  return (
    <div>
      <MatchRoute to="/posts/$postId" params={{ postId: "123" }} pending>
        {(match) => <Spinner show={!!match} wait="delay-50" />}
      </MatchRoute>
    </div>
  );
}
```
