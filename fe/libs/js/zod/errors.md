# отображение ошибок

Приоритет (от наивысшего):

- z.string("Not a string!");
- z.string().parse(12, {error: (iss) => "My custom error"});
- z.config({customError: (iss) => "My custom error"});
- z.config(z.locales.en());

Каждый результат проверки содержит

```ts
import * as z from "zod";

//{ success: false, error: ZodError }
const result = z.string("Not a string!").safeParse(12, {
  error: (iss) => {
    // можно обработать ошибку и здесь
  },
});
// объект
const result = z.string({ error: "some error message" }).safeParse(12);
//функция
const result = z.string({ error: () => "some error message" }).safeParse(12);
result.error.issues;
// [
//   {
//     expected: 'string',
//     code: 'invalid_type',
//     path: [],
//     message: "Not a string! <- текст ошибки переданной выше
//   }
// ]
```

# глобальная модификация ошибки

```ts
z.config({
  customError: (iss) => {
    return "globally modified error";
  },
});
```

# локализация ошибки

```ts
import * as z from "zod";
import { en } from "zod/locales";

z.config(en());
```

# утилиты для работы с ошибками

```ts
const tree = z.treeifyError(result.error); //развернет в дерево
const pretty = z.prettifyError(result.error); //читаемая ошибка
const flattened = z.flattenError(result.error); // { errors: string[], properties: { [key: string]: string[] } }
```
