# stripSearchParams

позволяет удалить параметры строки

```tsx
import { z } from "zod";
import { createFileRoute, stripSearchParams } from "@tanstack/react-router";
import { zodValidator } from "@tanstack/zod-adapter";

const defaultValues = {
  one: "abc",
  two: "xyz",
};

const searchSchema = z.object({
  one: z.string().default(defaultValues.one),
  two: z.string().default(defaultValues.two),
});

export const Route = createFileRoute("/")({
  validateSearch: zodValidator(searchSchema),
  search: {
    // strip default values
    middlewares: [stripSearchParams(defaultValues)],
  },
});
import { z } from "zod";
import { createRootRoute, stripSearchParams } from "@tanstack/react-router";
import { zodValidator } from "@tanstack/zod-adapter";

const searchSchema = z.object({
  hello: z.string().default("world"),
  requiredParam: z.string(),
});

export const Route = createRootRoute({
  validateSearch: zodValidator(searchSchema),
  search: {
    // always remove `hello`
    middlewares: [stripSearchParams(["hello"])],
  },
});
import { z } from "zod";
import { createFileRoute, stripSearchParams } from "@tanstack/react-router";
import { zodValidator } from "@tanstack/zod-adapter";

const searchSchema = z.object({
  one: z.string().default("abc"),
  two: z.string().default("xyz"),
});

export const Route = createFileRoute("/")({
  validateSearch: zodValidator(searchSchema),
  search: {
    // remove all search params
    middlewares: [stripSearchParams(true)],
  },
});
```
