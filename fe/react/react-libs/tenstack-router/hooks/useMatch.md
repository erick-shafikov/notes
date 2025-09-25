# useMatch

Параметры:

- Принимает (объект):
- - from - строка, если strict === true, То обязательный параметр
- - strict - boolean
- - select - (match: RouteMatch) => TSelected
- - structuralSharing - boolean для сравнение непримитивов возвращенных из select
- - shouldThrow - boolean при не нахождении пробросит ошибку

```ts
import { useMatch } from "@tanstack/react-router";

function Component() {
  const match = useMatch({ from: "/posts", shouldThrow: false });
  //     ^? RouteMatch | undefined
  if (match !== undefined) {
    // ...
  }
}
```

- Возвращает:
- - результат select
- - [RouteMatch](../types/RouteMatch.md)

```tsx
import { useMatch } from "@tanstack/react-router";

function Component() {
  const match = useMatch({ from: "/posts/$postId" });
  //     ^? strict match for RouteMatch
  // ...
}

import {
  useMatch,
  rootRouteId, // использование rootRouteId
} from "@tanstack/react-router";

function Component() {
  const match = useMatch({ from: rootRouteId });
  //     ^? strict match for RouteMatch
  // ...
}
```
