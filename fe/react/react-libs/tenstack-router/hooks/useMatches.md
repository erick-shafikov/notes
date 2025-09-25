# useMatches

Возвратит все RouteMatch объекты относительно позиционирования в дереве роутинга

Параметры:

- Принимает (объект):
- - select - (matches: RouteMatch[]) => TSelected
- - structuralSharing
- Возвращает:
- - [RouteMatch](../types/RouteMatch.md) вне зависимости от роута где вызывается

```tsx
import { useMatches } from "@tanstack/react-router";

function Component() {
  const matches = useMatches();
  //     ^? [RouteMatch, RouteMatch, ...]
  // ...
}
```
