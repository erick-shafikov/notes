# useQueries

```ts
const someArrayDataToFetch = [];

const {} = useQueries({
  queries: someArrayDataToFetch.map(({ id }) => ({
    queryKey: ["some", id],
    queryFn: () => fetchFunction(id),
  })),
});
```
