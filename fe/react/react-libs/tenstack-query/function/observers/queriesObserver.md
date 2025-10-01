# QueriesObserver

обзор за несколькими запросами

```ts
const observer = new QueriesObserver(queryClient, [
  { queryKey: ["post", 1], queryFn: fetchPost },
  { queryKey: ["post", 2], queryFn: fetchPost },
]);

const unsubscribe = observer.subscribe((result) => {
  console.log(result);
  unsubscribe();
});
```
