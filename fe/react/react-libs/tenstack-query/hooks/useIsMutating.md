# useIsMutating

количество мутаций

```ts
import { useIsMutating } from "@tanstack/react-query";
// How many mutations are fetching?
const isMutating = useIsMutating();
// How many mutations matching the posts prefix are fetching?
const isMutatingPosts = useIsMutating({ mutationKey: ["posts"] });
```
