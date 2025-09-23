# useChildMatches

Проверяет на соответствие все дочерние роуты

Параметры:

- параметры (объект с полями):
- - select (matches: [RouteMatch](../types/RouteMatch.md)[]) => TSelected
- - structuralSharing - boolean проверка объекта
- возвращает:
- select функцию или массив [RoteMatch](../types/RouteMatch.md)

```tsx
import { useChildMatches } from "@tanstack/react-router";

function Component() {
  const childMatches = useChildMatches();
  // ...
}
```
