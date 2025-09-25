# useSearch

Параметры:

- Принимает (объект):
- - from
- - shouldThrow
- - select
- - structuralSharing
- - strict
- Возвращает:
- - объект с параметрами поиска

```tsx
import { useSearch } from "@tanstack/react-router";

function Component() {
  const search = useSearch({ from: "/posts/$postId" });
  //    ^ FullSearchSchema

  // OR

  const selected = useSearch({
    from: "/posts/$postId",
    select: (search) => search.postView,
  });
  //    ^ string

  // OR

  const looseSearch = useSearch({ strict: false });
  //    ^ Partial<FullSearchSchema>

  // ...
}
```
