Функции проверки типов:

- Утилиты:
- - expectTypeOf
- - not
- - toEqualTypeOf

```ts
expectTypeOf({ a: 1 }).toEqualTypeOf({ a: 2 });
expectTypeOf({ a: 1, b: 1 }).not.toEqualTypeOf<{ a: number }>();
```

- - toMatchTypeOf

```ts
import { expectTypeOf } from "vitest";

expectTypeOf({ a: 1, b: 1 }).toMatchTypeOf({ a: 1 });
expectTypeOf<number>().toMatchTypeOf<string | number>();
expectTypeOf<string | number>().not.toMatchTypeOf<number>();
```

- - toExtend
- - toMatchObjectType
- - extract
- - exclude
- - returns
- - parameters
- - parameter
- - constructorParameters
- - instance
- - items
- - resolves
- - guards
- - asserts
- проверка на тип:
- - toBeAny

```ts
import { expectTypeOf } from "vitest";

expectTypeOf<any>().toBeAny();
expectTypeOf({} as any).toBeAny();
expectTypeOf("string").not.toBeAny();
```

- - toBeUnknown
- - toBeNever
- - toBeFunction
- - toBeObject
- - toBeArray
- - toBeString
- - toBeBoolean
- - toBeVoid
- - toBeSymbol
- - toBeNull
- - toBeUndefined
- - toBeNullable
- процесс:
- - toBeCallableWith
- - toBeConstructibleWith
- - toHaveProperty

# assertType

```ts
import { assertType } from "vitest";

function concat(a: string, b: string): string;
function concat(a: number, b: number): number;
function concat(a: string | number, b: string | number): string | number;

assertType<string>(concat("a", "b"));
assertType<number>(concat(1, 2));
// @ts-expect-error wrong types
assertType(concat("a", 2));
```
