# useInfiniteQuery

- если данные устареют то перезагрузку данных будет идти параллельно

```tsx
const {
  data: {
    pages: [],
    pageParams: {},
  },
  fetchNextPage,
  fetchPreviousPage,
  hasNextPage,
  hasPreviousPage,
  isFetchingNextPage,
  isFetchingPreviousPage,
} = useInfiniteQuery({
  initialPageParam: 0,
  getNextPageParam: (lastPage, pages) => lastPage.nextCursor,
  getPreviousPageParam: (firstPage, pages) => firstPage.prevCursor,
  // лимит по страницам
  maxPages: 3,
  // в случае если данные нужны в обратном порядке
  select: (data) => ({
    pages: [...data.pages].reverse(),
    pageParams: [...data.pageParams].reverse(),
  }),
});
```
