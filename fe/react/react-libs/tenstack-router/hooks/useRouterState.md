# useRouterState

Вернет внутреннее состояние роутера

Параметры:

- Принимает (объект):
- - select
- - structuralSharing
- Возвращает:
- - [RouterState](../types/RouterState.md)

```tsx
import { useRouterState } from "@tanstack/react-router";

function Component() {
  const state = useRouterState();

  const selected = useRouterState({
    select: (state) => state.location,
  });
}
```
