# useParentMatches

вернет все роуты выше не включая себя

Параметры:

- Принимает (объект):
- - select
- - structuralSharing
- Возвращает:
- - массив [RouteMatch](../types/RouteMatch.md)

```tsx
import { useParentMatches } from "@tanstack/react-router";

function Component() {
  const parentMatches = useParentMatches();
  //    ^ [RouteMatch, RouteMatch, ...]
}
```
