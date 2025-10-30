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

Утилита Custom

```ts
const px = z.custom<`${number}px`>((val) => {
  return typeof val === "string" ? /^\d+px$/.test(val) : false;
});

type px = z.infer<typeof px>; // `${number}px`

px.parse("42px"); // "42px"
px.parse("42vw"); // throws;
```

# Схемы

## примитивы

- z.string();
- - утилиты валидации строк:
- - - z.string().max(5);
- - - z.string().min(5);
- - - z.string().length(5);
- - - z.string().regex(/^[a-z]+$/);
- - - z.string().startsWith("aaa");
- - - z.string().endsWith("zzz");
- - - z.string().includes("---");
- - - z.string().uppercase();
- - - z.string().lowercase();
- - утилиты преобразования:
- - - z.string().trim();
- - - z.string().toLowerCase();
- - - z.string().toUpperCase();
- - - z.string().normalize();
- - валидация разных типов строк по содержанию для каждой предусмотрены доп настройки и расширения шаблона:
- - - email()
- - - uuid()
- - - url()
- - - httpUrl()
- - - hostname()
- - - emoji()
- - - base64()
- - - base64url()
- - - hex()
- - - jwt()
- - - nanoid()
- - - cuid()
- - - cuid2()
- - - ulid()
- - - ipv4()
- - - ipv6()
- - - cidrv4()
- - - cidrv6()
- - - hash("sha256") - "sha1", "sha384", "sha512", "md5"
- - - iso.date()
- - - iso.time()
- - - iso.datetime()
- - - iso.duration()
- - кастомные проверки:
- - - z.stringFormat
- - строки-шаблоны
- - - z.literal() - литеральный тип

      ```ts
      const colors = z.literal(["red", "green", "blue"]);
      colors.parse("green"); // ✅
      colors.parse("yellow"); // ❌
      ```

      ```ts
      const coolId = z.stringFormat("cool-id", () => {
        return val.length === 100 && val.startsWith("cool-");
      });
      const coolId = z.stringFormat("cool-id", /^cool-[a-z0-9]{95}$/);
      ```

- - - templateLiteral

      ```ts
      const schema = z.templateLiteral(["hello, ", z.string(), "!"]); // `hello, ${string}!
      ```

- z.number();
- - валидация значений:
- - - gt(5);
- - - gte(5); // alias .min(5)
- - - lt(5);
- - - lte(5); // alias .max(5)
- - - positive();
- - - nonnegative();
- - - negative();
- - - nonpositive();
- - - multipleOf(5);
- - z.nan().parse(NaN) - для NaN
- - z.int();
- - z.int32();
- - z.bigint();

- z.boolean();
- - Stringbools:
- - - z.stringbool().parse("true") true
- - - z.stringbool().parse("1") true
- - - z.stringbool().parse("yes") true
- - - z.stringbool().parse("on") true
- - - z.stringbool().parse("y") true
- - - z.stringbool().parse("enabled") true
- - - z.stringbool().parse("false"); false
- - - z.stringbool().parse("0"); false
- - - z.stringbool().parse("no"); false
- - - z.stringbool().parse("off"); false
- - - z.stringbool().parse("n"); false
- - - z.stringbool().parse("disabled"); false
- - - z.stringbool().parse(/_ anything else _/); // ZodError<[{ code: "invalid_value" }]>
- z.symbol();
- z.undefined();
- z.null();

## Принуждение:

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

## даты

```ts
z.date().min(new Date("1900-01-01"), { error: "Too old!" });
z.date().max(new Date(), { error: "Too young!" });
```

## Enums

```ts
const FishEnum = z.enum(["Salmon", "Tuna", "Trout"]);

FishEnum.parse("Salmon"); // => "Salmon"
FishEnum.parse("Swordfish"); // => ❌
```

Методы для создания новых enum:

- .exclude()
- .extract()

## Утилиты для типов

- z.optional
- z.nullable
- z.nullish
- z.any()
- z.unknown()
- z.never()

## Объекты

```ts
// all properties are required by default
const Person = z.object({
  name: z.string(),
  age: z.number(),
});

type Person = z.infer<typeof Person>;
// => { name: string; age: number; }
```

Разновидности:

- z.strictObject
- z.looseObject

Методы:

- .catchall() для определения типов которые не из объекта

  ```ts
  const DogWithStrings = z
    .object({
      name: z.string(),
      age: z.number().optional(),
    })
    .catchall(z.string());

  DogWithStrings.parse({ name: "Yeller", extraKey: "extraValue" }); // ✅
  DogWithStrings.parse({ name: "Yeller", extraKey: 42 }); // ❌
  ```

- .keyof()
- .extend()
- .safeExtend()
- .pick()
- .omit()
- .partial()
- .required()

Рекурсивные:

```ts
const Category = z.object({
  name: z.string(),
  get subcategories() {
    return z.array(Category);
  },
});

type Category = z.infer<typeof Category>;
// { name: string; subcategories: Category[] }
```

## Массивы

```ts
const stringArray = z.array(z.string()); // or z.string().array()

z.array(z.string()).min(5); // must contain 5 or more items
z.array(z.string()).max(5); // must contain 5 or fewer items
z.array(z.string()).length(5); // must contain 5 items exactly
```

## map, set

```ts
const StringNumberMap = z.map(z.string(), z.number());
type StringNumberMap = z.infer<typeof StringNumberMap>; // Map<string, number>

const myMap: StringNumberMap = new Map();
myMap.set("one", 1);
myMap.set("two", 2);

StringNumberMap.parse(myMap);
```

```ts
const NumberSet = z.set(z.number());
type NumberSet = z.infer<typeof NumberSet>; // Set<number>

const mySet: NumberSet = new Set();
mySet.add(1);
mySet.add(2);
NumberSet.parse(mySet);

z.set(z.string()).min(5); // must contain 5 or more items
z.set(z.string()).max(5); // must contain 5 or fewer items
z.set(z.string()).size(5); // must contain 5 items exactly
```

## TS-like

### Tuples

```ts
const MyTuple = z.tuple([z.string(), z.number(), z.boolean()]);

type MyTuple = z.infer<typeof MyTuple>;
// [string, number, boolean]
```

### Unions

```ts
const stringOrNumber = z.union([z.string(), z.number()]); // string | number

stringOrNumber.parse("foo"); // passes
stringOrNumber.parse(14); // passes
```

```ts
type MyResult =
  | { status: "success"; data: string }
  | { status: "failed"; error: string };

function handleResult(result: MyResult) {
  if (result.status === "success") {
    result.data; // string
  } else {
    result.error; // string
  }
}
```

### Intersections

```ts
const a = z.union([z.number(), z.string()]);
const b = z.union([z.number(), z.boolean()]);
const c = z.intersection(a, b);

type c = z.infer<typeof c>; // => number
```

### Records

```ts
const IdCache = z.record(z.string(), z.string());
type IdCache = z.infer<typeof IdCache>; // Record<string, string>

IdCache.parse({
  carlotta: "77d2586b-9e8e-4ecf-8b21-ea7e0530eadd",
  jimmie: "77d2586b-9e8e-4ecf-8b21-ea7e0530eadd",
});
```

### Branded types

```ts
type Cat = { name: string };
type Dog = { name: string };

const pluto: Dog = { name: "pluto" };
const simba: Cat = pluto; // works fine
```

### Readonly

пометить схему что она только для чтения

```ts
const ReadonlyUser = z.object({ name: z.string() }).readonly();
type ReadonlyUser = z.infer<typeof ReadonlyUser>;
// Readonly<{ name: string }>
```

## Files

```ts
const fileSchema = z.file();

z.file()
  .min(1)
  .max(1024 * 1024)
  .mime("image/png");

fileSchema.min(10_000); // minimum .size (bytes)
fileSchema.max(1_000_000); // maximum .size (bytes)
fileSchema.mime("image/png"); // MIME type
fileSchema.mime(["image/png", "image/jpeg"]); // multiple MIME types
```

## Instanceof

```ts
class Test {
  name: string;
}

const TestSchema = z.instanceof(Test);

TestSchema.parse(new Test()); // ✅
TestSchema.parse("whatever"); // ❌
```

## Property

```ts
const blobSchema = z
  .instanceof(URL)
  .check(
    z.property("protocol", z.literal("https:" as string, "Only HTTPS allowed"))
  );

blobSchema.parse(new URL("https://example.com")); // ✅
blobSchema.parse(new URL("http://example.com")); // ❌

const blobSchema = z.string().check(z.property("length", z.number().min(10)));

blobSchema.parse("hello there!"); // ✅
blobSchema.parse("hello."); // ❌
```

## JSON

```ts
const jsonSchema = z.lazy(() => {
  return z.union([
    z.string(params),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(jsonSchema),
    z.record(z.string(), jsonSchema),
  ]);
});
```

## Functions

```ts
const MyFunction = z.function({
  input: [z.string()], // parameters (must be an array or a ZodTuple)
  output: z.number(), // return type
});

type MyFunction = z.infer<typeof MyFunction>;
// (input: string) => number
```

# Кастомные схемы, Refinement

## .refine()

```ts
const myString = z.string().refine((val) => val.length <= 255, {
  error: "Too short!",
});

// цепочка
const myString = z
  .string()
  .refine((val) => val.length > 8, { error: "Too short!" })
  .refine((val) => val === val.toLowerCase(), { error: "Must be lowercase" });

// c объектом
const passwordForm = z
  .object({
    password: z.string(),
    confirm: z.string(),
  })
  .refine((data) => data.password === data.confirm, {
    message: "Passwords don't match",
    path: ["confirm"], // path of error
  });

// с объектом если основная проверка прошла
const schema = z
  .object({
    password: z.string().min(8),
    confirmPassword: z.string(),
    anotherField: z.string(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords do not match",
    path: ["confirmPassword"],

    // run if password & confirmPassword are valid
    when(payload) {
      return schema
        .pick({ password: true, confirmPassword: true })
        .safeParse(payload.value).success;
    },
  });

schema.parse({
  password: "asdf",
  confirmPassword: "asdf",
  anotherField: 1234, // ❌ this error will not prevent the password check from running
});

// работа с асинхронными функциями
const userId = z.string().refine(async (id) => {
  // verify that ID exists in database
  return true;
});
```

## .superRefine(), .check()

```ts
const UniqueStringArray = z.array(z.string()).superRefine((val, ctx) => {
  if (val.length > 3) {
    ctx.addIssue({
      code: "too_big",
      maximum: 3,
      origin: "array",
      inclusive: true,
      message: "Too many items 😡",
      input: val,
    });
  }

  if (val.length !== new Set(val).size) {
    ctx.addIssue({
      code: "custom",
      message: `No duplicates allowed.`,
      input: val,
    });
  }
});
```

# трансформации

## Codecs

[Подробнее](./codecs.md)

```ts
const stringToDate = z.codec(
  z.iso.datetime(), // input schema: ISO date string
  z.date(), // output schema: Date object
  {
    decode: (isoString) => new Date(isoString), // ISO string → Date
    encode: (date) => date.toISOString(), // Date → ISO string
  }
);

stringToDate.parse("2024-01-15T10:30:00.000Z"); // => Date
z.decode(stringToDate, "2024-01-15T10:30:00.000Z"); // => Date
z.encode(stringToDate, new Date("2024-01-15")); // => "2024-01-15T00:00:00.000Z"
```

## Pipes

```ts
const stringToLength = z.string().pipe(z.transform((val) => val.length));

stringToLength.parse("hello"); // => 5
```

## Transforms

[для преобразования в обе стороны](#codecs)

```ts
const castToString = z.transform((val) => String(val));

castToString.parse("asdf"); // => "asdf"
castToString.parse(123); // => "123"
castToString.parse(true); // => "true"

const coercedInt = z.transform((val, ctx) => {
  try {
    const parsed = Number.parseInt(String(val));
    return parsed;
  } catch (e) {
    ctx.issues.push({
      code: "custom",
      message: "Not a number",
      input: val,
    });

    // this is a special constant with type `never`
    // returning it lets you exit the transform without impacting the inferred return type
    return z.NEVER;
  }
});
```

## .transform()

```ts
const stringToLength = z.string().transform((val) => val.length);
```

## .preprocess()

```ts
const coercedInt = z.preprocess((val) => {
  if (typeof val === "string") {
    return Number.parseInt(val);
  }
  return val;
}, z.int());
```

# Утилиты

## Defaults, Prefaults

значение по умолчанию дял undefined

```ts
const defaultTuna = z.string().default("tuna");
defaultTuna.parse(undefined); // => "tuna"

const randomDefault = z.number().default(Math.random);
randomDefault.parse(undefined); // => 0.4413456736055323
randomDefault.parse(undefined); // => 0.1871840107401901
randomDefault.parse(undefined); // => 0.7223408162401552

const schema = z
  .string()
  .transform((val) => val.length)
  .default(0);
schema.parse(undefined); // => 0
```

## Catch

Возврат при ошибке

```ts
const numberWithCatch = z.number().catch(42);

numberWithCatch.parse(5); // => 5
numberWithCatch.parse("tuna"); // => 42
```

# метаданные

Можно определить метаданные для экземпляра zod

```ts
import * as z from "zod";
//регистрация
const myRegistry = z.registry<{ description: string }>();

const mySchema = z.string();
//регистрация
myRegistry.add(mySchema, { description: "A cool schema!" });
myRegistry.has(mySchema); // => true
myRegistry.get(mySchema); // => { description: "A cool schema!" }
myRegistry.remove(mySchema);
myRegistry.clear(); // wipe registry
```

## register

определение глобальных полей

```ts
import * as z from "zod";

//вариант 2 через register функцию
const mySchema = z.string();

mySchema.register(myRegistry, { description: "A cool schema!" });

const emailSchema = z.email().register(z.globalRegistry, {
  id: "email_address",
  title: "Email address",
  description: "Your email address",
  examples: ["first.last@example.com"],
});

//ts стандартные
export interface GlobalMeta {
  id?: string;
  title?: string;
  description?: string;
  deprecated?: boolean;
  [k: string]: unknown;
}

declare module "zod" {
  interface GlobalMeta {
    // add new fields here
    examples?: unknown[];
  }
}
```

## .meta()

```ts
const emailSchema = z.email().meta({
  id: "email_address",
  title: "Email address",
  description: "Please enter a valid email address",
});

emailSchema.meta();
// => { id: "email_address", title: "Email address", ... }
```

## .describe()

```ts
const emailSchema = z.email();
emailSchema.describe("An email address");

// equivalent to
emailSchema.meta({ description: "An email address" });
```
