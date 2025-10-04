# useIsFetching

вернет количество загружаемых запросов

```ts
import { useIsFetching } from "@tanstack/react-query";
// How many queries are fetching?
const isFetching = useIsFetching();
// How many queries matching the posts prefix are fetching?
const isFetchingPosts = useIsFetching({ queryKey: ["posts"] });
```
