# QueryObserver

принимает те же опции что и [useQuery](../hooks/useQuery.md)

```ts
const observer = new QueryObserver(queryClient, { queryKey: ["posts"] });

const unsubscribe = observer.subscribe((result) => {
  console.log(result);
  unsubscribe();
});
```
