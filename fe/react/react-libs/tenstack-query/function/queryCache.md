# queryCache

Хранит весь кеш (данные, мета-информацию, состояние запросов) tsq

```ts
import { QueryCache } from "@tanstack/react-query";

const queryCache = new QueryCache({
  onError: (error) => {
    console.log(error);
  },
  onSuccess: (data) => {
    console.log(data);
  },
  onSettled: (data, error) => {
    console.log(data, error);
  },
});

const query = queryCache.find(["posts"]);
```

# конструктор

принимает на вход объект с полями:

- onError?: (error: unknown, query: Query) => void

- onSuccess?: (data: unknown, query: Query) => void

- onSettled?: (data: unknown | undefined, error: unknown | null, query: Query) => void

# методы экземпляр

- find - поиск по ключу
- findAll
- subscribe - подписка на изменения
- clear
