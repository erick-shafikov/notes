# useMutationState

Для доступа в кеш мутаций

```ts
import { useMutationState } from "@tanstack/react-query";

const mutation = useMutation({
  mutationKey,
  mutationFn: (newPost) => {
    return axios.post("/posts", newPost);
  },
});

// Array<TResult>
const variables = useMutationState(
  {
    //два параметра
    filters: { status: "pending" },
    select: (mutation) => mutation.state.variables,
  },
  // QueryClient
  queryClient
);
```
