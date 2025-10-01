# mutationCache

```ts
import { MutationCache } from "@tanstack/react-query";

const mutationCache = new MutationCache({
  onError: (error) => {
    console.log(error);
  },
  onSuccess: (data) => {
    console.log(data);
  },
});
```

конструктор принимает объект с полями:

- onError

```js
// пример с 401
const queryClient = new QueryClient({
  onError: (error) => {
    if (error.status === 401) {
      localStorage.removeItem("token");
    }
  },
});
```

- onSuccess
- onSettled
- onMutate

Методы экземпляра:

- getAll
- subscribe

```ts
const callback = (event) => {
  // onError, onSuccess, onSettled and onMutate события
  console.log(event.type, event.mutation);
};

const unsubscribe = mutationCache.subscribe(callback);
```

- clear
