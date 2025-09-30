# InfiniteQueryObserver

принимает те же параметры что и [useInfiniteQuery](../hooks/useInfiniteQuery.md)

```ts
const observer = new InfiniteQueryObserver(queryClient, {
  queryKey: ["posts"],
  queryFn: fetchPosts,
  getNextPageParam: (lastPage, allPages) => lastPage.nextCursor,
  getPreviousPageParam: (firstPage, allPages) => firstPage.prevCursor,
});

const unsubscribe = observer.subscribe((result) => {
  console.log(result);
  unsubscribe();
});
```
