# retainSearchParams

Позволяет сохранить параметры поиска. Если строка запроса нужна в других роутах

```tsx
import { z } from "zod";
import { createRootRoute, retainSearchParams } from "@tanstack/react-router";
import { zodValidator } from "@tanstack/zod-adapter";

const searchSchema = z.object({
  rootValue: z.string().optional(),
});

export const Route = createRootRoute({
  validateSearch: zodValidator(searchSchema),
  search: {
    middlewares: [retainSearchParams(["rootValue"])],
  },
});
import { z } from "zod";
import { createFileRoute, retainSearchParams } from "@tanstack/react-router";
import { zodValidator } from "@tanstack/zod-adapter";

const searchSchema = z.object({
  one: z.string().optional(),
  two: z.string().optional(),
});

export const Route = createFileRoute("/")({
  validateSearch: zodValidator(searchSchema),
  search: {
    middlewares: [retainSearchParams(true)],
  },
});
```
