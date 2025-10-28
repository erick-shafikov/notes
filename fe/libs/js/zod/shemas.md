# определение схемы

```ts
import * as z from "zod";

const Player = z.object({
  username: z.string(),
  xp: z.number(),
});

// обработка с помощью try/catch
try {
  Player.parse({ username: "billie", xp: 100 });
  await Player.parseAsync({ username: "billie", xp: 100 });
} catch (e) {
  if (error instanceof z.ZodError) {
    // доступ к тексту ошибки
    error.issues;
    /* [
      {
        expected: 'string',
        code: 'invalid_type',
        path: [ 'username' ],
        message: 'Invalid input: expected string'
      },
      {
        expected: 'number',
        code: 'invalid_type',
        path: [ 'xp' ],
        message: 'Invalid input: expected number'
      }
    ] */
  }
}

// через safeParse
const result = Player.safeParse({ username: 42, xp: "100" });
if (!result.success) {
  result.error; // ZodError instance
} else {
  result.data; // { username: string; xp: number }
}
```

# извлечение типов

```ts
const Player = z.object({
  username: z.string(),
  xp: z.number(),
});

// extract the inferred type { username: "billie", xp: 100 }
type Player = z.infer<typeof Player>;
```

С использованием transform

```ts
//с использованием transform
const mySchema = z.string().transform((val) => val.length);

type MySchemaIn = z.input<typeof mySchema>;
// => string

type MySchemaOut = z.output<typeof mySchema>; // z.infer<typeof mySchema>
// number
```

# Схемы

## примитивы

- z.string();
- - z.literal()

```ts
const colors = z.literal(["red", "green", "blue"]);

colors.parse("green"); // ✅
colors.parse("yellow"); // ❌
```

- z.number();
- z.bigint();
- z.boolean();
- z.symbol();
- z.undefined();
- z.null();

Принуждение:

- z.coerce.string() - String(input)
- z.coerce.number() - Number(input)
- z.coerce.boolean() - Boolean(input)
- z.coerce.bigint() - BigInt(input)

```ts
//применение
const schema = z.coerce.string();

schema.parse("tuna"); // => "tuna"
schema.parse(42); // => "42"
schema.parse(true); // => "true"
schema.parse(null); // => "null"

//вывод типов
const A = z.coerce.number();
type AInput = z.input<typeof A>; // => unknown

const B = z.coerce.number<number>();
type BInput = z.input<typeof B>; // => number
```
